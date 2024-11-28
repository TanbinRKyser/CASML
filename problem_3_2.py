from pybloom_live import BloomFilter
from collections import defaultdict

def get_frequent_elements(lst_of_elements):
    """This function uses your proposed Bloomfilter strategy to get the items occuring four or more times.

    Args:
        lst_of_elements (iterable, generator): dataset containing items with different frequencies

    Returns:
        items_four_or_more (list): Resulting frequernt items that occur for or more times in the dataset
    """
    ### TODO: Add/change code below
    bloom = BloomFilter(capacity=len(lst_of_elements), error_rate=0.01)

    item_counts = defaultdict(int)

    for item in lst_of_elements:
        if item not in bloom:
            bloom.add(item)  
        item_counts[item] += 1

    items_four_or_more = [item for item, count in item_counts.items() if count >= 4]
    print(len(items_four_or_more))

    ### TODO: Add/change code above
    return items_four_or_more


if __name__ == "__main__":
    ### NOTE: The main clause will not be graded  
    ### TODO: Change to individual path
    path_to_dataset = "dataset_1M.csv"
    with open(path_to_dataset, "r") as f:
        dataset = [line.strip() for line in f]
    result = get_frequent_elements(dataset)
    print(result)
