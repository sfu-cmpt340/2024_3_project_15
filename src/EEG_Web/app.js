document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('eegFile');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Assuming a backend API endpoint exists to handle the uploaded file
    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Display the result from the backend
        document.getElementById('result').innerHTML = `<h3>Analysis Result:</h3><p>${data.result}</p>`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = '<p style="color: red;">Error occurred while uploading and analyzing the EEG signal.</p>';
    });
});