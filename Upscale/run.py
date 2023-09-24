from globals import config, hyperparameters
from ExperimentLogger import ExperimentLogger
import sys
experimentlogger = None

def main():

    if len(sys.argv) < 2:
        raise Exception("You must provide at least one argument: 'prepare', 'train', or 'test'.")

    action = sys.argv[1]

    if action == 'prepare':
        from Data.DataUtils import prepare_files
        if len(sys.argv) < 4:
            raise Exception("You must provide paths for feature_pack_dir and label_pack_dir.")
        feature_pack_dir = sys.argv[2]
        label_pack_dir = sys.argv[3]
        prepare_files( feature_pack_dir=feature_pack_dir, label_pack_dir=label_pack_dir, resolution=config()["resolution"])

    elif action == 'train':
        from ResNetImplementation.PerformTraining import PerformTraining
        global experimentlogger
        experimentlogger= ExperimentLogger()
        PerformTraining(config(), hyperparameters())

    elif action == 'testmod':
        from Data.DataUtils import extract_png_tensors_from_jar
        if len(sys.argv) < 4:
            raise Exception("You must provide a path to the .jar file for the mod and run name.")
        jar_file = sys.argv[2]
        run_name = sys.argv[3]
        tensors=extract_png_tensors_from_jar(jar_file)
        exp_logger=ExperimentLogger.from_existing_run(run_name)
        exp_logger.get_snapshot(tensors)

    else:
        raise Exception(f"Invalid argument: {action}. Only 'prepare', 'train', or 'test' are allowed.")

if __name__ == "__main__":
    import sys

    print(sys.path)
    print('inside')
    main()



"""
vanilla_textures: "/Users/gametekker/Downloads/VanillaDefault+1.20",
highres_textures: "/Users/gametekker/Downloads/Faithful+64x+-+Beta+9",
"""