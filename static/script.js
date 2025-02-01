document.getElementById("upload-form").addEventListener("submit", function (e) {
    e.preventDefault();
    
    let fileInput = document.getElementById("file-input").files[0];
    if (!fileInput) {
        alert("Please select a CSV file.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);

    document.getElementById("status").innerText = "Processing...";

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById("status").innerText = "Processing Complete!";
            document.getElementById("download-link").style.display = "block";
        } else {
            document.getElementById("status").innerText = "Error processing file.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("status").innerText = "Upload failed!";
    });
});
