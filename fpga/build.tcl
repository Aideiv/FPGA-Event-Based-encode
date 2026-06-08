# build.tcl — Vitis HLS Synthesis & Export Script
# 
# Synthesizes the collision_avoidance_top pipeline for:
#   - Zynq UltraScale+ MPSoC (XCZU9EG)
#   - Kria K26 SOM
#
# Usage:
#   vitis_hls -f build.tcl                    # Synthesize + export IP
#   vitis_hls -f build.tcl csim              # C simulation only
#   vitis_hls -f build.tcl synth             # Synthesis only
#   vitis_hls -f build.tcl cosim             # Co-simulation with Vivado
#
# Prerequisites:
#   - Vitis HLS 2023.1+ or Vivado HLS 2020.2+
#   - Xilinx license for the target part

# ── Project Setup ────────────────────────────────────────────────────────────
open_project collision_avoidance
set_top collision_avoidance_top

# Add design files
add_files top_level.cpp -cflags "-I."
add_files aer_interface.h
add_files ring_buffer.h
add_files normalization.h
add_files spatial_hash.h
add_files encoder_systolic.h
add_files pwm_output.h

# Add testbench
add_files -tb testbench.cpp -cflags "-I."

# ── Solution Configuration ───────────────────────────────────────────────────
open_solution "solution_zcu9eg" -flow_target vitis

# Target part: Zynq UltraScale+ MPSoC XCZU9EG-2FFVB1156
set_part {xczu9eg-ffvb1156-2-i}

# Clock constraint: 100MHz (10ns period)
create_clock -period 10 -name clk

# ── Synthesis Directives ─────────────────────────────────────────────────────

# Top-level: AXI4-Lite control interface
set_directive_interface -mode s_axilite -bundle CTRL "collision_avoidance_top"
set_directive_interface -mode s_axilite -bundle CTRL "collision_avoidance_top" ctrl_regs
set_directive_interface -mode s_axilite -bundle DEBUG "collision_avoidance_top" debug_flow

# Data interfaces (BRAM/stream for high-throughput paths)
set_directive_interface -mode bram "collision_avoidance_top" events
set_directive_interface -mode bram "collision_avoidance_top" knn_results
set_directive_interface -mode bram "collision_avoidance_top" events_txy
set_directive_interface -mode bram "collision_avoidance_top" flow_pred
set_directive_interface -mode bram "collision_avoidance_top" flow_uncert
set_directive_interface -mode bram "collision_avoidance_top" weights

# Pipeline performance targets
# Main pipeline: 1 event per clock throughput
set_directive_pipeline -II 1 "collision_avoidance_top/COLLECT_EVENTS"
set_directive_pipeline -II 1 "collision_avoidance_top/NORM_LOOP"
set_directive_pipeline -II 1 "collision_avoidance_top/FLOW_AGGREGATE"

# k-NN spatial hash: parallel search across 3×3 cells
set_directive_pipeline -II 1 "spatial_hash_knn/BUCKET_EVENTS"
set_directive_pipeline -II 1 "spatial_hash_knn/KNN_LOOP"
set_directive_pipeline -II 1 "spatial_hash_knn/SEARCH_CELLS"
set_directive_loop_flatten "spatial_hash_knn/SEARCH_CELLS"

# Encoder: systolic array parallelism
set_directive_pipeline -II 1 "local_geometry_encoder/STAGE1_PA"
set_directive_pipeline -II 1 "local_geometry_encoder/STAGE2_TRIG"
set_directive_pipeline -II 1 "local_geometry_encoder/STAGE3_SPARSE"
set_directive_pipeline -II 1 "local_geometry_encoder/STAGE4_OUTPUT"
set_directive_unroll -factor 8 "local_geometry_encoder/tile_matmul" d
set_directive_unroll -factor 4 "local_geometry_encoder/STAGE2_TRIG" d
set_directive_unroll -factor 4 "local_geometry_encoder/SPARSE_ACCUM" d
set_directive_unroll -factor 8 "local_geometry_encoder/STAGE4_OUTPUT" d

# CORDIC: fully unrolled (8 iterations)
set_directive_unroll "cordic_sin_cos" i

# Ring buffer: dual-port BRAM configuration
set_directive_array_partition -type cyclic -factor 4 -dim 1 "ring_buffer" buffer

# ── Optimization Strategy ────────────────────────────────────────────────────

# Dataflow: enable task-level parallelism between pipeline stages
# (Not used due to shared BRAM arrays — stages are sequential by design)
# set_directive_dataflow "collision_avoidance_top"

# Resource allocation: balance BRAM vs LUT trade-off
# Spatial hash grid: partition for parallel access
set_directive_array_partition -type cyclic -factor 8 -dim 1 "spatial_hash_knn" grid
set_directive_bind_storage -type ram_t2p -impl bram "spatial_hash_knn" grid

# Encoder arrays: partition for DSP parallelism
set_directive_array_partition -type cyclic -factor 16 -dim 2 "local_geometry_encoder" pA
set_directive_array_partition -type cyclic -factor 8 -dim 2 "local_geometry_encoder" epA_real
set_directive_array_partition -type cyclic -factor 8 -dim 2 "local_geometry_encoder" epA_imag
set_directive_array_partition -type cyclic -factor 8 -dim 2 "local_geometry_encoder" G_real
set_directive_array_partition -type cyclic -factor 8 -dim 2 "local_geometry_encoder" G_imag

# Encoder weights: ROM (read-only, loaded at init)
set_directive_bind_storage -type rom_t2p -impl bram "local_geometry_encoder" enc_weights

# ── Build Targets (select via command line argument) ──────────────────────────

# Determine which step to run
if { $argc >= 1 } {
    set TARGET [lindex $argv 0]
} else {
    set TARGET "synth"
}

switch $TARGET {
    "csim" {
        # C simulation (verify functional correctness)
        puts "Running C simulation..."
        csim_design -clean
    }
    "synth" {
        # Synthesis only (generate resource/performance reports)
        puts "Running synthesis..."
        csynth_design
    }
    "cosim" {
        # Co-simulation with Vivado (verify RTL matches C)
        puts "Running C/RTL co-simulation..."
        cosim_design -trace_level all -tool xsim
    }
    "export" {
        # Export as packaged IP for Vivado block design
        puts "Exporting IP..."
        export_design -format ip_catalog -vendor "Enotrium" -version "1.0.0"
    }
    default {
        puts "Usage: vitis_hls -f build.tcl \[csim|synth|cosim|export\]"
        puts "  csim  — C simulation (functional verification)"
        puts "  synth — Synthesis (resource/latency reports)"
        puts "  cosim — C/RTL co-simulation (RTL verification)"
        puts "  export — Export IP for Vivado integration"
        exit 1
    }
}

puts "\n============================================"
puts "  FPGA synthesis complete: $TARGET"
puts "  Reports in: collision_avoidance/solution_zcu9eg/syn/report/"
puts "============================================"