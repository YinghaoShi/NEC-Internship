
process_mnist_data.py:
	
	usage: 
		process_mnist() is for processing mnist to the pkl file with the structure of 
			dict{"training_images", 
				"test_images", 
				"training_labels", 
				"test_labels"}

		Please check the save_path in __main__
		For Emnist and Fashion:
			Please check and modify the path of Emnist_filename and Fashion_filename (the default path should be fine).
		For Mnist:
			It will automatically download and process the mnist data.

		Simpily run:    python process_mnist_data.py should work.


After getting the pkl file, all data could be directly used to create the task data using the following function.

For creating sequential task data:
	
	Please use create_task_data.py inside create_task_data folder.
	
	Specify the number of task, data path, save path, data name (fashion or emnist or mnist), operation_type (rotation or permutation).
	
	Usage example:
		
		python create_task_data.py --data_path ../Data/EMNIST/emnist.pkl --save_path ../../Data/EMNIST --data_name emnist --operation_type permutation





Inside the Vis.ipynb:
	Be careful with the data path.


	

	