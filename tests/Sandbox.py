# Your list of tuples
results = [
    ('apple', 5, '(_red'), 
    ('banana', 2, '12yellow'), 
    ('cherry', 8, '[5_g(reen'), 
    ('date', 3, 'brown')]

# Your custom sorting sequence
sort_sequence = ['r', 'b', 'y', 'g']

# Define a sorting function that extracts the first alphabetic letter from the second element of each tuple
def custom_sort_key(item):
    second_element = str(item[2])  # Convert the second element to a string
    first_alpha_letter = next((char for char in second_element if char.isalpha()), '')
    return sort_sequence.index(first_alpha_letter) if first_alpha_letter in sort_sequence else len(sort_sequence)

# Sort the results list based on the custom sorting key
sorted_results = sorted(results, key=custom_sort_key)

print(sorted_results)
