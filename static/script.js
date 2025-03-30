
document.getElementById("submit_button").addEventListener("click",function(event){
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById("img_selected");

    if(fileInput.files.length===0){
        document.getElementById("error_message").textContent = "*Please upload a file before submitting";
    }
    else{
        document.getElementById("error_message").textContent = "";     
    }

    formData.append("image",fileInput.files[0]);

    fetch("http://localhost:5000/predict",{
        method: "POST",
        body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
        if(data.prediction === 0){
            document.getElementById("predicted_output").textContent="Cat";
        }
        else if(data.prediction === 1){
            document.getElementById("predicted_output").textContent="Dog";
        }
    })
    .catch((error) => {
        console.error("Error:",error);
    });
    
});
