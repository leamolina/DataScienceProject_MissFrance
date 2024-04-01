import random
import streamlit as st
import time
import matplotlib.pyplot as plt


# README :
# Before executing the program, make sure you have installed the package streamlit
# If not, write in the terminal : pip install streamlit
# To execute the program (with the interface), go on your terminal and write : streamlit run Sort.py


# We decide to implement three different sorting algorithms : MergeSort, InsertionSort & BubbleSort
# We will decide which one is the best, based on their running time.

# We know the worst case running times of these sorting algorithms:
# Merge Sort : O(n log n)
# Insertion Sort : O(n^2)
# Bubble Sort : O(n^2)


# 1. Merge Sort

def merge(array1, array2):
    merged_array = []
    i = 0
    j = 0

    while i < len(array1) and j < len(array2):
        if array1[i] <= array2[j]:
            merged_array.append(array1[i])
            i += 1
        else:
            merged_array.append(array2[j])
            j += 1

    while i < len(array1):
        merged_array.append(array1[i])
        i += 1
    while j < len(array2):
        merged_array.append(array2[j])
        j += 1

    return merged_array


def mergeSort(arr):
    if len(arr) == 1:
        return arr

    half = len(arr) // 2
    left_array = arr[:half]
    right_array = arr[half:]

    left_array = mergeSort(left_array)
    right_array = mergeSort(right_array)

    return merge(left_array, right_array)


# 2. Insertion sort

def insertionSort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j - 1] > arr[j]:
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
            j -= 1
    return arr


# 3. Bubble Sort

def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr



# Bonus : increasing order :

def merge_decreasing(array1, array2):
    merged_array = []
    i = 0
    j = 0

    while i < len(array1) and j < len(array2):
        if array1[i] >= array2[j]:
            merged_array.append(array1[i])
            i += 1
        else:
            merged_array.append(array2[j])
            j += 1

    while i < len(array1):
        merged_array.append(array1[i])
        i += 1
    while j < len(array2):
        merged_array.append(array2[j])
        j += 1

    return merged_array

def mergeSort_decreasing(arr):
    if len(arr) == 1:
        return arr

    half = len(arr) // 2
    left_array = arr[:half]
    right_array = arr[half:]

    left_array = mergeSort_decreasing(left_array)
    right_array = mergeSort_decreasing(right_array)

    return merge_decreasing(left_array, right_array)

def insertionSort_decreasing(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j - 1] < arr[j]:
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
            j -= 1
    return arr

def bubbleSort_decreasing(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def runningTime(f, arr):
    time_start = time.time()
    f(arr)
    time_end = time.time()
    return time_end - time_start


# Implementation of an Interface :

st.title("Comparison of different sorting algorithms")
price_min = st.number_input("Minimal price", value=0.0, min_value=0.0, max_value=1000000.0, step=0.01, format="%f")
price_max = st.number_input("Maximal price", value=0.0, min_value=0.0, max_value=1000000.0, step=0.01, format="%f")
array_len = st.number_input("Length of the list", value=1, min_value=1, max_value=1000000)

if st.button("Generate random list of price"):
    array_prices = [round(random.uniform(price_min, price_max), 2) for i in range(array_len)]

    st.write("List of random prices : ")
    st.write(array_prices)

    # MergeSort
    st.subheader("Merge Sort")
    st.write("Sorted array with Merge Sort", mergeSort(array_prices))
    st.write("Sorted array with Merge Sort (increasing order) ", mergeSort_decreasing(array_prices))
    running_time_merge = runningTime(mergeSort, array_prices)
    st.write("Running time of Merge Sort :", running_time_merge, "s")

    # InsertionSort
    st.subheader("Insertion Sort")
    st.write("Sorted array with Insertion Sort", insertionSort(array_prices))
    st.write("Sorted array with Insertion Sort (decreasing order) ", insertionSort_decreasing(array_prices))
    running_time_insertion = runningTime(insertionSort, array_prices)
    st.write("Running time of Insertion Sort :", running_time_insertion, "s")

    # BubbleSort
    st.subheader("Bubble Sort")
    st.write("Sorted array with Bubble Sort", bubbleSort(array_prices))
    st.write("Sorted array with Bubble Sort (decreasing order) ", bubbleSort_decreasing(array_prices))
    running_time_bubble = runningTime(bubbleSort, array_prices)
    st.write("Running time of Bubble Sort :", running_time_bubble, "s")

    if st.button("Choose other inputs"):
        stop = True

    # Plot:
    st.subheader("Plot")
    array_sizes = [20, 60, 100, 200, 300, 500]
    insertion_times = []
    merge_times = []
    bubble_times = []
    # Filling the running time arrays
    for size in array_sizes:
        array = [round(random.uniform(price_min, price_max), 2) for i in range(size)]
        insertion_times.append(runningTime(insertionSort, array))
        merge_times.append(runningTime(mergeSort, array))
        bubble_times.append(runningTime(bubbleSort, array))
    # Graph:
    fig, ax = plt.subplots()
    ax.plot(array_sizes, merge_times, label="Merge Sort", color="purple")
    ax.plot(array_sizes, insertion_times, label="Insertion Sort", color="green")
    ax.plot(array_sizes, bubble_times, label="Bubble Sort", color="pink")
    ax.set_xlabel("Array size")
    ax.set_ylabel("Running time (in second) ")
    ax.set_title("Comparison of the running times")
    ax.legend()
    st.pyplot(fig)
