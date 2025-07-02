import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
SUMMARY = []

def convert_model(model_name, saved_model_dir):
    tfjs_output_dir = os.path.join(os.path.dirname(saved_model_dir), "tfjs_model")
    os.makedirs(tfjs_output_dir, exist_ok=True)
    # Conditionally add quantization for sentiment model
    cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        saved_model_dir,
        tfjs_output_dir,
    ]
    # Quantize only sentiment model
    if model_name == "sentiment":
        cmd += ["--quantize_float16", "*"]
    print(f"🔄 Converting [{model_name}] to tfjs...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ [{model_name}] converted to {tfjs_output_dir}")
        SUMMARY.append((model_name, "Converted", tfjs_output_dir))
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR converting [{model_name}]:\n{e.stderr.decode()}")
        SUMMARY.append((model_name, "Failed", str(e)))

def main():
    print(f"🔍 Scanning '{ROOT}' for models...")
    found_any = False
    for model_name in os.listdir(ROOT):
        model_dir = os.path.join(ROOT, model_name)
        saved_model_dir = os.path.join(model_dir, "saved_model")
        saved_model_pb = os.path.join(saved_model_dir, "saved_model.pb")
        if os.path.isdir(saved_model_dir) and os.path.isfile(saved_model_pb):
            found_any = True
            convert_model(model_name, saved_model_dir)
        elif os.path.isdir(saved_model_dir):
            print(f"⚠️  {model_name}: Directory exists but 'saved_model.pb' missing. Skipping.")
            SUMMARY.append((model_name, "Skipped (no saved_model.pb)", ""))
        else:
            SUMMARY.append((model_name, "Skipped (no saved_model)", ""))
    if not found_any:
        print("⚠️  No models with 'saved_model' found in outputs/. Nothing to convert.")
    print("\n====== Conversion Summary ======")
    for model, status, path in SUMMARY:
        print(f"{model.ljust(20)} | {status.ljust(18)} | {path}")
    print("================================\n")

if __name__ == "__main__":
    main()
