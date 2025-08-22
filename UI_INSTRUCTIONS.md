# 🦗 Insect Classifier UI

A simple web interface to test your trained insect classifier model using Gradio.

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install gradio joblib
   ```

2. **Launch the UI**:
   ```bash
   python app.py
   ```

3. **Open your browser** to: http://localhost:7860

## What You Can Do

- **Upload Audio Files**: Drag and drop .wav files of insect sounds
- **Get Predictions**: See the top predicted species with confidence scores
- **View All Species**: Check the collapsible "Supported Species" section to see all 12 trained species
- **Visual Results**: Bar chart showing confidence scores for the top predictions

## Supported Species

Your model can currently identify these 12 insect species:

1. Aphidoletes aphidimyza
2. Bombus terrestris  
3. Bradysia difformis
4. Coccinella septempunctata
5. Episyrphus balteatus
6. Halyomorpha halys
7. Myzus persicae
8. Nezara viridula
9. Palomena prasina
10. Rhaphigaster nebulos
11. Trialeurodes vaporariorum
12. Tuta absoluta

## Audio Requirements

- **Format**: WAV files work best
- **Duration**: Audio will be automatically trimmed/padded to 2.5 seconds
- **Sample Rate**: Automatically resampled to 16kHz
- **Quality**: Clear recordings with minimal background noise work best

## Troubleshooting

### UI won't start
- Make sure you have gradio installed: `pip install gradio`
- Check that your model file exists: `models/trained/cnn_lstm_best.pth`
- Verify the label encoder exists: `models/trained/label_encoder.joblib`

### Low prediction confidence
- Try different audio files
- Ensure audio is clear and contains insect sounds
- Check that the species is one of the 12 supported ones

### Model not found error
- Make sure you've trained the model first: `python scripts/train_model.py`
- Check that training completed successfully and saved the model

## Performance Notes

- **Accuracy**: Currently achieving ~71% accuracy on the validation set
- **Speed**: Predictions are nearly instantaneous on CPU
- **GPU**: Will automatically use GPU if available for faster inference

## Next Steps

This UI is perfect for:
- Testing your current 12-species model
- Validating model performance with real audio samples
- Demonstrating the classifier to others
- Preparing for expansion to the 459-species dataset

Ready to expand to more species? The current model provides a solid foundation for transfer learning to larger datasets!