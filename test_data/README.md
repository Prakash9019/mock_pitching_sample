# Test Data Directory

Place your test audio files (.wav or .mp3) in this directory for testing the API.

## Example Usage

```bash
# Test with a sample audio file
python test_api.py test_data/sample_pitch.mp3
```

## Creating Test Audio

If you don't have a sample audio file, you can create one using various tools:

### Using PowerShell (Windows)

```powershell
# Record 10 seconds of audio
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()
$builder = New-Object System.Speech.Recognition.GrammarBuilder
$builder.AppendDictation()
$grammar = New-Object System.Speech.Recognition.Grammar($builder)
$recognizer.LoadGrammar($grammar)
$recognizer.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
Start-Sleep -Seconds 10
$recognizer.RecognizeAsyncStop()
$result = $recognizer.Recognize()
$result.Audio.WriteToWaveFile("test_data/sample_pitch.wav")
```

### Using Online Tools

You can also use online text-to-speech tools to create sample audio files for testing.