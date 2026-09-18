geometry_msgs/Point center  # bbox center [px] in image frame (x,y), z=0
geometry_msgs/Vector3 size  # bbox size [px] (x=width, y=height), z=0
float32 confidence          # [0..1]
uint8 color                # use the enum below

# Constants
uint8 BLUE=0
uint8 RED=1
uint8 YELLOW_BIG=2
uint8 YELLOW_SMALL=3
uint8 UNKNOWN=4
