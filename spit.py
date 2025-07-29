import os
import random

def split_train_test_fixed_projects(t01_train_file, t01_test_file, interaction_file, train_output, test_output, removal_percentage):
    """
    Split the interaction data into train and test files, maintaining the project range of t01 and applying a percentage removal on TPLs.

    Args:
    - t01_train_file: Path to the t01 train.txt to get project IDs.
    - t01_test_file: Path to the t01 test.txt to get project IDs.
    - interaction_file: Path to the interaction.txt file.
    - train_output: Path to save the train.txt file.
    - test_output: Path to save the test.txt file.
    - removal_percentage: Percentage of TPLs to remove for the test set.
    """
    # Load the t01 train and test files to get the exact project IDs and row count
    t01_train_projects = []
    t01_test_projects = []

    with open(t01_train_file, 'r') as f:
        t01_train_projects = [line.split()[0] for line in f.readlines()]
    
    with open(t01_test_file, 'r') as f:
        t01_test_projects = [line.split()[0] for line in f.readlines()]

    # Read the full interaction.txt file
    project_interactions = {}
    with open(interaction_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            project_id = parts[0]
            libraries = parts[1:]
            project_interactions[project_id] = libraries

    train_data = []
    test_data = []

    # Create the train and test files for the specific range
    for project_id in t01_train_projects:
        if project_id in project_interactions:
            libraries = project_interactions[project_id]
            num_libraries = len(libraries)
            num_remove = int(num_libraries * removal_percentage)

            # Randomly sample libraries for test
            test_libraries = random.sample(libraries, num_remove)
            train_libraries = [lib for lib in libraries if lib not in test_libraries]

            # Ensure at least 1 library in both train and test
            if len(train_libraries) == 0:
                train_libraries = test_libraries[:1]
                test_libraries = test_libraries[1:]

            train_data.append(f"{project_id} {' '.join(train_libraries)}\n")
            test_data.append(f"{project_id} {' '.join(test_libraries)}\n")

    for project_id in t01_test_projects:
        if project_id in project_interactions:
            libraries = project_interactions[project_id]
            num_libraries = len(libraries)
            num_remove = int(num_libraries * removal_percentage)

            # Randomly sample libraries for test
            test_libraries = random.sample(libraries, num_remove)
            train_libraries = [lib for lib in libraries if lib not in test_libraries]

            # Ensure at least 1 library in both train and test
            if len(train_libraries) == 0:
                train_libraries = test_libraries[:1]
                test_libraries = test_libraries[1:]

            train_data.append(f"{project_id} {' '.join(train_libraries)}\n")
            test_data.append(f"{project_id} {' '.join(test_libraries)}\n")

    # Save the generated train.txt and test.txt
    with open(train_output, 'w') as f_train:
        f_train.writelines(train_data)

    with open(test_output, 'w') as f_test:
        f_test.writelines(test_data)

    print(f"Train and test files created: {train_output}, {test_output}")


# Paths to t01 train.txt, test.txt and interaction.txt
t01_train_file = 'datasets/PyLib/t01/train.txt'
t01_test_file = 'datasets/PyLib/t01/test.txt'
interaction_file = 'datasets/Original/interaction.txt'

# t02 (20% removal)
t02_train_output = 'datasets/PyLib/t02/train.txt'
t02_test_output = 'datasets/PyLib/t02/test.txt'
split_train_test_fixed_projects(t01_train_file, t01_test_file, interaction_file, t02_train_output, t02_test_output, removal_percentage=0.2)

# t04 (40% removal)
t04_train_output = 'datasets/PyLib/t04/train.txt'
t04_test_output = 'datasets/PyLib/t04/test.txt'
split_train_test_fixed_projects(t01_train_file, t01_test_file, interaction_file, t04_train_output, t04_test_output, removal_percentage=0.4)

# t06 (60% removal)
t06_train_output = 'datasets/PyLib/t06/train.txt'
t06_test_output = 'datasets/PyLib/t06/test.txt'
split_train_test_fixed_projects(t01_train_file, t01_test_file, interaction_file, t06_train_output, t06_test_output, removal_percentage=0.6)

