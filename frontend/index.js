const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('file', form.file.files[0]);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to classify image');
        }

        const result = await response.json();
        resultDiv.textContent = `Predicted Food: ${result.prediction}`;
    } catch (error) {
        console.error('Error:', error);
        resultDiv.textContent = 'Error classifying image';
    }
});
