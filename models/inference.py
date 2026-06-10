import importlib.resources as pkg_resources
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from . import models  # Import the models subpackage
from .estimator import NormalEstimator

class NormalFlowEstimator(nn.Module):
    def __init__(self, training_set='UNION', auto_scale_time=True, use_knn=False, k_neighbors=32):
        """ Initialize the normal flow estimator.
        Args:
            training_set (str): The training set used to train the model. 
                There are four options: 'UNION', 'MVSEC', 'DSEC', 'EVIMO'.
                Use 'FPGA' for the FPGA-optimized d=128 parameter preset.
            auto_scale_time (bool): 
                If True, the input time will be multiplied with an estimated time scale to try best fitting 50000 events per step.
                If False, the input time will be used as is.
            use_knn (bool):
                If True, use O(n·k) k-NN spatial hashing for adjacency.
                If False, use O(n²) full pairwise distance matrix.
                Default False for backward compatibility.
            k_neighbors (int):
                Number of nearest neighbors when use_knn=True. Default 32.
        """
        valid_sets = ['UNION', 'EVIMO', 'DSEC', 'MVSEC', 'FPGA']
        assert training_set in valid_sets, f"Invalid training set: {training_set}"
        
        if training_set == 'FPGA':
            from .params import FPGAParams as P
            self.use_knn = True
            self.k_neighbors = P.k_neighbors
        elif training_set == 'UNION':
            from .params import UNIONParams as P
            self.use_knn = use_knn
            self.k_neighbors = k_neighbors
        elif training_set == 'EVIMO':
            from .params import EVIMOParams as P
            self.use_knn = use_knn
            self.k_neighbors = k_neighbors
        elif training_set == 'DSEC':
            from .params import DSECParams as P
            self.use_knn = use_knn
            self.k_neighbors = k_neighbors
        elif training_set == 'MVSEC':
            from .params import MVSECParams as P
            self.use_knn = use_knn
            self.k_neighbors = k_neighbors
        else:
            raise ValueError("Invalid training set")
        self.P = P

        super(NormalFlowEstimator, self).__init__()
        self.estimator = self.init_model()
        self.load_model(self.estimator, training_set)
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.estimator = self.estimator.cuda()
            
        self.auto_scale_time = auto_scale_time
        self.ring_buffer_capacity = getattr(self.P, 'ring_buffer_size', 4096)
        self.ring_buffer_events_xy = torch.zeros(0, 2)
        self.ring_buffer_events_t = torch.zeros(0)
        self.ring_buffer_write_ptr = 0
    
    @staticmethod
    def load_model(estimator, training_set):
        """Load the appropriate weight file for the training set.
        
        For 'FPGA' training_set, loads FPGA.pth (d=128, alpha=8).
        This file is generated from UNION.pth by train/convert_weights_to_fpga.py.
        If FPGA.pth is missing, falls back to UNION.pth with a warning.
        """
        import warnings
        load_key = training_set
        try:
            with pkg_resources.open_binary(models, f"{load_key}.pth") as f:
                state_dict = torch.load(f, map_location=torch.device("cpu"), weights_only=True)
        except FileNotFoundError:
            if training_set == 'FPGA':
                # FPGA.pth not generated yet — fall back to UNION and hope shapes match
                warnings.warn(
                    "FPGA.pth not found. Falling back to UNION.pth. "
                    "Run: python train/convert_weights_to_fpga.py --input models/models/UNION.pth "
                    "--output models/models/FPGA.pth"
                )
                with pkg_resources.open_binary(models, "UNION.pth") as f:
                    state_dict = torch.load(f, map_location=torch.device("cpu"), weights_only=True)
            else:
                raise
        estimator.load_state_dict(state_dict)
        estimator.eval()
        return estimator
    
    def init_model(self):
        return NormalEstimator(
            self.P.d, self.P.alpha, 
            use_knn=self.use_knn, 
            k_neighbors=self.k_neighbors
        )
    
    def inference(self, events_t, events_xy, ensemble=3):
        assert len(events_t.shape) == 1 and len(events_xy.shape) == 2, "Invalid input shape"
        assert events_t.shape[0] == events_xy.shape[0], "Inconsistent input shape"
        assert torch.all(events_t[1:] >= events_t[:-1]), "Events are not sorted"
        
        events_t = events_t / self.P.t_radius
        events_t = (events_t - events_t.min()).float()
        events_xy = events_xy / self.P.pxl_radius
        flow_predictions = torch.zeros(events_t.shape[0], 2)
        flow_uncertainty = torch.zeros(events_t.shape[0]) + 9999
        
        if self.auto_scale_time:
            time_scale = self.time_scale_mining(events_t)
        else:
            time_scale = 1.0
        scaled_events_t = events_t * time_scale
        progress_bar = tqdm(
            total=events_t.shape[0], 
            desc=f"computing per-event normal flow"
        )
        for start, end, events_txy in self.slice_events(scaled_events_t, events_xy):
            with torch.no_grad():
                if self.cuda_available:
                    events_txy = events_txy.cuda()
                flow_pred, flow_uncert = self.estimator.inference(
                    events_txy, ensemble=ensemble)
                flow_predictions[start:end] = flow_pred * time_scale
                flow_uncertainty[start:end] = flow_uncert
            progress_bar.update((end-start).item())
        progress_bar.close()

        return flow_predictions, flow_uncertainty
    
    def inference_fpga(self, events_t, events_xy):
        """ FPGA-optimized inference with ring buffer and single-pass mode.
        
        This is the recommended path for FPGA deployment:
          - Uses get_knn_adjacency() (O(n·k)) instead of get_adj_matrix() (O(n²))
          - Uses inference_single_pass() (no ensemble rotation)
          - Maintains a ring buffer of fixed-size events for real-time operation
          - Returns (flow_pred, flow_uncert) tensors
        
        Args:
            events_t: (n,) tensor of timestamps
            events_xy: (n, 2) tensor of (x, y) coordinates
        
        Returns:
            flow_pred: (n, 2) tensor of (vx, vy) flow predictions
            flow_uncert: (n,) tensor of uncertainty values (zeros for single-pass)
        """
        assert len(events_t.shape) == 1 and len(events_xy.shape) == 2, "Invalid input shape"
        assert events_t.shape[0] == events_xy.shape[0], "Inconsistent input shape"
        
        # Normalize coordinates to unit range
        events_t = events_t / self.P.t_radius
        events_t = (events_t - events_t.min()).float()
        events_xy = events_xy / self.P.pxl_radius
        
        if self.auto_scale_time:
            time_scale = self.time_scale_mining(events_t)
        else:
            time_scale = 1.0
        scaled_events_t = events_t * time_scale
        
        flow_predictions = torch.zeros(events_t.shape[0], 2)
        flow_uncertainty = torch.zeros(events_t.shape[0])
        
        # Fixed-size window slicing (no dynamic allocation)
        for start, end, events_txy in self.slice_events_ring_buffer(
            scaled_events_t, events_xy, self.ring_buffer_capacity
        ):
            n_events = events_txy.shape[0]
            if n_events < 3:
                continue
            with torch.no_grad():
                if self.cuda_available:
                    events_txy = events_txy.cuda()
                flow_pred, flow_uncert = self.estimator.inference_single_pass(events_txy)
                flow_predictions[start:end] = flow_pred * time_scale
                flow_uncertainty[start:end] = flow_uncert
        
        return flow_predictions, flow_uncertainty
    
    def push_event_to_ring_buffer(self, t, x, y):
        """ Push a single event into the online ring buffer.
        
        For real-time FPGA operation: events stream in continuously.
        The ring buffer keeps the latest self.ring_buffer_capacity events.
        When full, oldest events are overwritten.
        
        Args:
            t: scalar timestamp
            x: scalar x coordinate
            y: scalar y coordinate
        
        Returns:
            ready: bool, True if buffer has enough events for inference
        """
        self.ring_buffer_write_ptr = (self.ring_buffer_write_ptr + 1) % self.ring_buffer_capacity
        
        if self.ring_buffer_events_xy.shape[0] < self.ring_buffer_capacity:
            # Buffer growing
            self.ring_buffer_events_xy = torch.cat([
                self.ring_buffer_events_xy,
                torch.tensor([[x, y]])
            ])
            self.ring_buffer_events_t = torch.cat([
                self.ring_buffer_events_t,
                torch.tensor([t])
            ])
        else:
            # Buffer full, overwrite
            idx = self.ring_buffer_write_ptr
            self.ring_buffer_events_xy[idx] = torch.tensor([x, y])
            self.ring_buffer_events_t[idx] = t
        
        # Signal ready when buffer is at least half full
        return self.ring_buffer_events_t.shape[0] >= self.ring_buffer_capacity // 2
    
    def flush_ring_buffer(self):
        """ Run inference on the current ring buffer contents.
        
        Returns:
            flow_pred: (m, 2) tensor of flow predictions for m≤ring_buffer_capacity events
            flow_uncert: (m,) tensor of uncertainties
        """
        n = self.ring_buffer_events_t.shape[0]
        if n < 3:
            return torch.zeros(0, 2), torch.zeros(0)
        
        events_t_norm = self.ring_buffer_events_t / self.P.t_radius
        events_t_norm = (events_t_norm - events_t_norm.min()).float()
        events_xy_norm = self.ring_buffer_events_xy / self.P.pxl_radius
        
        if self.auto_scale_time:
            scale = self.time_scale_mining(events_t_norm)
        else:
            scale = 1.0
        
        events_t_scaled = events_t_norm * scale
        events_txy = torch.cat([events_t_scaled[:, None], events_xy_norm], dim=-1)
        
        with torch.no_grad():
            if self.cuda_available:
                events_txy = events_txy.cuda()
            flow_pred, flow_uncert = self.estimator.inference_single_pass(events_txy)
        
        return flow_pred * scale, flow_uncert
        
    def slice_events(self, events_t, events_xy):
        t_min, t_max = events_t.min(), events_t.max()
        num_steps = int((t_max - t_min) // 2)                          # after normalizing, the step size will be 2*1=2
        if num_steps < 2:
            num_steps = 2
        t_grid = np.linspace(
            t_min, 
            t_min + num_steps * 2, 
            num_steps, False)
        for i in range(num_steps-1):
            start, end = t_grid[i], t_grid[i+1]
            start_idx = torch.searchsorted(events_t, start)
            end_idx = torch.searchsorted(events_t, end)
            if start_idx >= end_idx:
                continue
            events = torch.cat([
                events_t[start_idx:end_idx, None], 
                events_xy[start_idx:end_idx]], dim=-1)
            yield start_idx, end_idx, events
    
    def slice_events_ring_buffer(self, events_t, events_xy, capacity):
        """ Fixed-size window slicing — maps directly to HLS ring buffer.
        
        Unlike slice_events() which uses variable-sized time-based windows,
        this uses fixed-size spatial windows that map to a BRAM ring buffer.
        No dynamic memory allocation.
        
        Args:
            events_t: (n,) normalized timestamp tensor
            events_xy: (n, 2) normalized coordinate tensor
            capacity: int, maximum events per window (ring buffer depth)
        
        Yields:
            (start_idx, end_idx, events_txy) tuples with events_txy.shape[0] <= capacity
        """
        n = events_t.shape[0]
        for start in range(0, n, capacity):
            end = min(start + capacity, n)
            events = torch.cat([
                events_t[start:end, None],
                events_xy[start:end]
            ], dim=-1)
            yield start, end, events
        
    def time_scale_mining(self, events_t):
        """ Estimate the time scale coefficient for normalizing the timestamps.
        After normalizing the timestamps, the number of events per step should be around 50000.
        """
        t_min, t_max = events_t.min(), events_t.max()
        num_steps = int((t_max - t_min) // 2)                          # after normalizing, the step size will be 2*1=2
        if num_steps < 2:
            return 0.5
        t_grid = np.linspace(
            t_min, 
            t_min + num_steps * 2, 
            num_steps, False)
        num_events_per_step = np.zeros(num_steps-1)
        for i in range(num_steps-1):
            start, end = t_grid[i], t_grid[i+1]
            start_idx = torch.searchsorted(events_t, start)
            end_idx = torch.searchsorted(events_t, end)
            num_events_per_step[i] = end_idx - start_idx
        num_events_per_step = np.median(num_events_per_step)
        scale = num_events_per_step / 50000.
        scale = np.clip(scale, 0.33, 1.33)
        return scale