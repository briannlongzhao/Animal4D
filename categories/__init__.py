category_ids = {
    "deer": 0,
    "fox": 1,
    "horse": 2,
    "boar": 3,
    "moose": 4,
    "bear": 5,
    "cow": 6,
    "wolf": 7,
    "rabbit": 8,
    "panther": 9,
    "leopard": 10,
    "elephant": 11,
    "tiger": 12,
    "racoon": 13,
    "cougar": 14,
    "goat": 15,
    "pig": 16,
    "sheep": 17,
    "hippo": 18,
    "rhino": 19,
    "zebra": 20,
    "cat": 21,
    "dog": 22,
}


SMAL_categories = {
    0: "tiger",
    1: "dog",
    2: "horse",
    3: "cow",
    4: "hippo",
}


# Input label name output category index
def get_label_idx(label):
    return category_ids.get(label, -1)


# Input category index output label name
def get_idx_label(idx):
    return {v: k for k, v in category_ids.items()}.get(idx, None)


# Input label or index output index and name
def process_category(cat):
    try:
        index = int(cat)
        animal = get_idx_label(index)
    except ValueError:
        animal = cat.strip("\'\"").lower()
        index = get_label_idx(animal)
    return index, animal


# Process list of indices or labels
def process_categories(cats):
    indices, animals = [], []
    for cat in cats:
        index, animal = process_category(cat)
        indices.append(index)
        animals.append(animal)
    return indices, animals
