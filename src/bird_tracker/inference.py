import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path=args.model_file)
