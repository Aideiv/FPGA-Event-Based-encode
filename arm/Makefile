# arm/Makefile — Build System for ARM C++ Collision Avoidance Code
#
# Targets:
#   make              : Build drone_control binary (bare-metal or Linux)
#   make sim           : Build with simulator (no FPGA hardware needed)
#   make freertos      : Build for FreeRTOS on Zynq R5
#   make clean         : Remove build artifacts

# ---------------------------------------------------------------------------
# Compiler selection
# ---------------------------------------------------------------------------
# For Zynq ARM Linux (Cortex-A53):
CXX ?= arm-linux-gnueabihf-g++
# For bare-metal Zynq R5:
# CXX = armr5-none-eabi-g++

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
CXXFLAGS_COMMON = -std=c++17 -O3 -Wall -Wextra -march=armv8-a -mcpu=cortex-a53

# Debug build
DEBUG_FLAGS = -g -O0 -DDEBUG

# Release build  
RELEASE_FLAGS = -O3 -DNDEBUG -ffast-math -funroll-loops

CXXFLAGS = $(CXXFLAGS_COMMON) $(RELEASE_FLAGS)

# Simulation mode (no FPGA hardware needed, links fake register interface)
SIM_FLAGS = -DSIMULATION -DFPGA_SIM

# FreeRTOS (for Cortex-R5 bare-metal)
FREERTOS_FLAGS = -DFREERTOS -ffreestanding -nostdlib
FREERTOS_PATH ?= /opt/FreeRTOSv10.5.1

# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
INCLUDES = -I. -I$(FREERTOS_PATH)/include -I$(FREERTOS_PATH)/portable/GCC/ARM_CR5

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
TARGET = drone_control
TARGET_SIM = drone_control_sim

# Source files (header-only for collision_predictor.h, evasion_controller.h)
SOURCES = drone_main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# ---------------------------------------------------------------------------
# Build rules
# ---------------------------------------------------------------------------
.PHONY: all sim freertos clean

# Default: hardware target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ -lpthread -lrt
	@echo "Built $@ for Zynq ARM (hardware mode)"

# Simulation target (no FPGA hardware needed)
sim: CXXFLAGS += $(SIM_FLAGS)
sim: $(TARGET_SIM)

$(TARGET_SIM): $(OBJECTS)
	g++ $(CXXFLAGS) $(INCLUDES) -o $@ $^ -lpthread
	@echo "Built $@ (simulation mode — no FPGA required)"

# FreeRTOS bare-metal target (Cortex-R5)
freertos: CXXFLAGS += $(FREERTOS_FLAGS)
freertos: INCLUDES += -I$(FREERTOS_PATH)/include -I$(FREERTOS_PATH)/portable/GCC/ARM_CR5
freertos: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET)_rtos.elf $^ \
		$(FREERTOS_PATH)/portable/GCC/ARM_CR5/port.o \
		$(FREERTOS_PATH)/tasks.o \
		$(FREERTOS_PATH)/queue.o \
		$(FREERTOS_PATH)/list.o \
		-T zynq_r5_linker.ld
	@echo "Built $(TARGET)_rtos.elf for FreeRTOS"

# Compile source files
%.o: %.cpp collision_predictor.h evasion_controller.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(TARGET_SIM) $(TARGET)_rtos.elf *.o
	@echo "Clean complete"

# ---------------------------------------------------------------------------
# Flash / deploy targets
# ---------------------------------------------------------------------------
# Deploy to Zynq via SCP (adjust IP address)
deploy:
	scp $(TARGET) root@192.168.1.10:/usr/local/bin/
	@echo "Deployed to Zynq"

# Run on remote Zynq
run:
	ssh root@192.168.1.10 "/usr/local/bin/$(TARGET)"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "ARM Collision Avoidance Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build binary for Zynq ARM Linux"
	@echo "  make sim      Build simulation binary (run on desktop, no FPGA needed)"
	@echo "  make freertos Build bare-metal binary for Zynq R5 with FreeRTOS"
	@echo "  make clean    Remove build artifacts"
	@echo "  make deploy   SCP binary to Zynq board"
	@echo "  make run      SSH into Zynq and run binary"
	@echo ""
	@echo "Variables:"
	@echo "  CXX           Compiler (default: arm-linux-gnueabihf-g++)"
	@echo "  FREERTOS_PATH FreeRTOS source path (default: /opt/FreeRTOSv10.5.1)"