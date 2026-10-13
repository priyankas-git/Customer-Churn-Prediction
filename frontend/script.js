document.addEventListener("DOMContentLoaded", () => {
  const churnForm = document.getElementById("churnForm");
  const predictBtn = document.getElementById("predictBtn");
  const resultContainer = document.getElementById("resultContainer");
  const errorContainer = document.getElementById("errorContainer");
  const predictionText = document.getElementById("predictionText");
  const probabilityText = document.getElementById("probabilityText");
  const errorMessage = document.getElementById("errorMessage");

  churnForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Clear previous results/errors
    resultContainer.classList.add("hidden");
    errorContainer.classList.add("hidden");
    errorMessage.textContent = "";

    // Read form values
    const payload = {
      tenure: parseFloat(document.getElementById("tenure").value),
      monthly_charges: parseFloat(document.getElementById("monthlyCharges").value),
      total_charges: parseFloat(document.getElementById("totalCharges").value),
      contract_type: document.getElementById("contractType").value,
      internet_service: document.getElementById("internetService").value,
      tech_support: document.getElementById("techSupport").value,
    };

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // Display prediction
      if (data.prediction === 1 || data.churn === true) {
        predictionText.textContent = "Likely to Churn";
        predictionText.className = "churn-yes";
      } else {
        predictionText.textContent = "Likely to Stay";
        predictionText.className = "churn-no";
      }

      // Display probability formatted as percentage
      const pct = (data.probability * 100).toFixed(1);
      probabilityText.textContent = `${pct}%`;

      resultContainer.classList.remove("hidden");
    } catch (err) {
      errorMessage.textContent = err.message || "Failed to make prediction.";
      errorContainer.classList.remove("hidden");
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict Churn";
    }
  });
});
