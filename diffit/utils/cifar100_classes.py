"""
CIFAR-100 Class Mappings and Utilities

CIFAR-100 has 100 fine classes grouped into 20 superclasses.
"""

# CIFAR-100 fine class names (0-99)
CIFAR100_FINE_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# CIFAR-100 superclass names (0-19)
CIFAR100_SUPERCLASSES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 
    'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
    'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

# Mapping from fine classes to superclasses
FINE_TO_SUPERCLASS = {
    # aquatic_mammals
    4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
    # fish
    1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
    # flowers
    54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
    # food_containers
    9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
    # fruit_and_vegetables
    0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
    # household_electrical_devices
    22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
    # household_furniture
    5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
    # insects
    6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
    # large_carnivores
    3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
    # large_man-made_outdoor_things
    12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
    # large_natural_outdoor_scenes
    23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
    # large_omnivores_and_herbivores
    15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
    # medium_mammals
    34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
    # non-insect_invertebrates
    26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
    # people
    2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
    # reptiles
    27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
    # small_mammals
    36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
    # trees
    47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
    # vehicles_1
    8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
    # vehicles_2
    41: 19, 69: 19, 81: 19, 85: 19, 89: 19
}

# Superclass to fine classes mapping
SUPERCLASS_TO_FINE = {}
for fine_class, superclass in FINE_TO_SUPERCLASS.items():
    if superclass not in SUPERCLASS_TO_FINE:
        SUPERCLASS_TO_FINE[superclass] = []
    SUPERCLASS_TO_FINE[superclass].append(fine_class)

# Predefined class groups for common filtering scenarios
PREDEFINED_GROUPS = {
    'animals': [3, 4, 6, 7, 14, 15, 18, 19, 21, 24, 26, 27, 29, 30, 31, 34, 36, 38, 42, 43, 44, 45, 50, 55, 63, 64, 65, 66, 72, 74, 75, 77, 78, 79, 80, 88, 93, 95, 97, 99],
    'vehicles': [8, 13, 41, 48, 58, 69, 81, 85, 89, 90],
    'nature': [23, 33, 47, 49, 52, 54, 56, 59, 60, 62, 70, 71, 82, 92, 96],
    'household': [5, 9, 10, 16, 20, 22, 25, 28, 39, 40, 61, 84, 86, 87, 94],
    'food': [0, 51, 53, 57, 83],
    'people': [2, 11, 35, 46, 98],
    'structures': [12, 17, 37, 68, 76],
    'trees': [47, 52, 56, 59, 96]  # maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
}


def get_class_indices(class_specification):
    """
    Convert class specification to list of class indices.
    
    Args:
        class_specification: Can be:
            - List of integers (class indices)
            - List of strings (class names)
            - String (predefined group name or superclass name)
    
    Returns:
        List of class indices (0-99)
    """
    if isinstance(class_specification, list):
        if all(isinstance(x, int) for x in class_specification):
            # List of indices
            return [x for x in class_specification if 0 <= x <= 99]
        elif all(isinstance(x, str) for x in class_specification):
            # List of class names
            indices = []
            for name in class_specification:
                if name in CIFAR100_FINE_CLASSES:
                    indices.append(CIFAR100_FINE_CLASSES.index(name))
            return indices
    elif isinstance(class_specification, str):
        # Check predefined groups
        if class_specification in PREDEFINED_GROUPS:
            return PREDEFINED_GROUPS[class_specification]
        
        # Check superclass names
        if class_specification in CIFAR100_SUPERCLASSES:
            superclass_idx = CIFAR100_SUPERCLASSES.index(class_specification)
            return SUPERCLASS_TO_FINE.get(superclass_idx, [])
        
        # Check if it's a single class name
        if class_specification in CIFAR100_FINE_CLASSES:
            return [CIFAR100_FINE_CLASSES.index(class_specification)]
    
    return []


def filter_dataset_by_classes(dataset, target_classes, relabel=True):
    """
    Filter dataset to include only specified classes.
    
    Args:
        dataset: List of (image_tensor, label) tuples or just image tensors
        target_classes: List of class indices to keep
        relabel: If True, relabel classes from 0 to len(target_classes)-1
    
    Returns:
        Filtered dataset and class mapping (if relabeled)
    """
    if not target_classes:
        return dataset, None
    
    target_classes = sorted(target_classes)
    
    # Create class mapping for relabeling
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(target_classes)}
    
    filtered_data = []
    
    # Handle different dataset formats
    for item in dataset:
        if isinstance(item, tuple) and len(item) == 2:
            # (image, label) format
            image, label = item
            if label in target_classes:
                if relabel:
                    new_label = class_mapping[label]
                    filtered_data.append((image, new_label))
                else:
                    filtered_data.append((image, label))
        else:
            # Image-only format (for diffusion models)
            # We need to get the label from somewhere - this would need to be handled
            # in the dataset loading function where we have access to labels
            filtered_data.append(item)
    
    return filtered_data, class_mapping if relabel else None


def get_class_info(class_indices):
    """Get human-readable information about selected classes."""
    info = {
        'count': len(class_indices),
        'classes': [],
        'superclasses': set()
    }
    
    for idx in sorted(class_indices):
        if 0 <= idx <= 99:
            class_name = CIFAR100_FINE_CLASSES[idx]
            superclass_idx = FINE_TO_SUPERCLASS.get(idx)
            superclass_name = CIFAR100_SUPERCLASSES[superclass_idx] if superclass_idx is not None else 'unknown'
            
            info['classes'].append({
                'index': idx,
                'name': class_name,
                'superclass': superclass_name
            })
            info['superclasses'].add(superclass_name)
    
    return info
