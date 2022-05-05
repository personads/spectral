import logging, os, re, sys


def setup_experiment(out_path, prediction=False):
	if not os.path.exists(out_path):
		if prediction:
			print(f"Experiment path '{out_path}' does not exist. Cannot run prediction. Exiting.")
			exit(1)

		# if output dir does not exist, create it (new experiment)
		print(f"Path '{out_path}' does not exist. Creating...")
		os.mkdir(out_path)
	# if output dir exist, check if predicting
	else:
		# if not predicting, verify overwrite
		if not prediction:
			response = None

			while response not in ['y', 'n']:
				response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
			if response == 'n':
				exit(1)

	# setup logging
	log_format = '%(message)s'
	log_level = logging.INFO
	logging.basicConfig(filename=os.path.join(out_path, 'classify.log'), filemode='a', format=log_format, level=log_level)

	logger = logging.getLogger()
	logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_filter(filter_id):
	from models.encoders import PrismEncoder

	# set up filter definitions
	filter_match = re.match(r'(?P<name>[a-z]+?)(?P<args>\((\d+,? *)*\))', filter_id, flags=re.IGNORECASE)
	args_pattern = re.compile(r'(\d+),?')

	# check if filter syntax is correct
	if filter_match:
		filter_args = [int(a) for a in args_pattern.findall(filter_match['args'])]
		# parse nofilter
		if filter_match['name'] == 'nofilter':
			if len(filter_args) != 0:
				logging.error(f"[Error] nofilter does not take any arguments. Exiting.")
				exit(1)
			return None
		# parse eqalloc filter
		elif filter_match['name'] == 'eqalloc':
			if len(filter_args) != 3:
				logging.error(f"[Error] eqalloc filter requires three arguments 'max_freqs', 'num_bands', 'band_idx'. Exiting.")
				exit(1)
			return PrismEncoder.gen_equal_allocation_filters(filter_args[0], filter_args[1])[filter_args[2]]
		# parse band filter
		elif filter_match['name'] == 'band':
			if len(filter_args) != 3:
				logging.error(f"[Error] band filter requires three arguments 'max_freqs', 'start_idx', 'end_idx'. Exiting.")
				exit(1)
			return PrismEncoder.gen_band_filter(filter_args[0], filter_args[1], filter_args[2])
		elif filter_match['name'] == 'auto':
			if len(filter_args) != 1:
				logging.error(f"[Error] auto requires the argument 'max_freqs'. Exiting.")
				exit(1)
			return PrismEncoder.gen_auto_filter_initialization(filter_args[0])
		else:
			logging.error(f"[Error] Unknown filter type '{filter_match['name']}'. Exiting.")
			exit(1)
	else:
		logging.error(f"[Error] Could not parse filter '{filter_id}'. Exiting.")
		exit(1)
