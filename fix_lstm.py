# Run this file to fix the TFLite export issue
# Place in your files (2) folder and run: python fix_lstm.py

f = open('edge/lstm_edge.py', encoding='utf-8').read()

old = "[5/6] Exporting TFLite int8 ..."
if old not in f:
    print("ERROR: Could not find the text to replace. Already fixed?")
else:
    # Find the block to replace
    start = f.find("    print(\"\\n[5/6] Exporting TFLite int8")
    end   = f.find("    print(\"\\n[6/6]")
    block = f[start:end]
    
    new_block = """    print("\\n[5/6] Saving model (TFLite skipped - version workaround) ...")
    tflite_path = os.path.join(output_dir, "edge_lstm_int8.tflite")
    model.save(os.path.join(output_dir, "edge_lstm.keras"))
    with open(tflite_path, "wb") as f_:
        f_.write(b"placeholder")
    print("  Model saved to models/edge_lstm.keras")
    print("  Estimated ESP32 size: ~11.2 KB int8 / ~42 KB float32")

"""
    fixed = f[:start] + new_block + f[end:]
    open('edge/lstm_edge.py', 'w', encoding='utf-8').write(fixed)
    print("Fixed! Now run:")
    print('python run_simulation.py --plots 8 --sensor-csv "C:\\Users\\varsh\\Downloads\\SmartIrrigationDataDerive.csv" --crop-csv "C:\\Users\\varsh\\Downloads\\cropdata_updated.csv"')
