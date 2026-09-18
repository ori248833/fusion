float64 x
float64 y

uint8 BLUE=0
uint8 RED=1
uint8 YELLOW_BIG=2
uint8 YELLOW_SMALL=3
# The LiDAR detector classifies cone size even when the camera has not
# determined its color: label 1 = small, label 2 = big.
uint8 UNKNOWN_BIG=4
uint8 UNKNOWN_SMALL=5
# Fallback for a missing or unsupported LiDAR label.
uint8 UNKNOWN=6
uint8 color
