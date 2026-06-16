# ===========================================================================
# Configuration file for the Car Designing System
# ===========================================================================

# Path where the dataset images are stored
DATASET_PATH = r"data\car_dataset"

# Number of top visually similar images to retrieve from a text prompt
TOP_K = 5

# Default text prompt for the search box
TEXT_PROMPT = "A sleek red sports car with aerodynamic design and low profile"

# Which Vision-Language model to use ("clip" is standard)
PREFERRED_MODEL = "clip"

# These are the car categories the system will use when analyzing and labeling the images
CAR_CATEGORIES = [
    "SUV",
    "sedan",
    "sports car",
    "pickup truck",
    "convertible",
    "coupe",
    "hatchback",
    "minivan",
    "muscle car",
    "luxury car",
    "electric car",
]
