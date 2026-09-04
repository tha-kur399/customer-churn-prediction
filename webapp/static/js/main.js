/**
 * Customer Churn Prediction System - Frontend Interactions & Demo Presets
 */

function updateTotalChargesAuto() {
    const tenureEl = document.getElementById('tenure');
    const monthlyEl = document.getElementById('MonthlyCharges');
    const totalEl = document.getElementById('TotalCharges');

    if (tenureEl && monthlyEl && totalEl) {
        const tenure = parseFloat(tenureEl.value) || 1;
        const monthly = parseFloat(monthlyEl.value) || 20;
        totalEl.value = (tenure * monthly).toFixed(2);
    }
}

function applyPreset(presetType) {
    const form = document.getElementById('churnForm');
    if (!form) return;

    if (presetType === 'high_risk') {
        // High Risk Profile: Month-to-month, new customer, Fiber optic, No security/tech support, Electronic check
        form.gender.value = 'Female';
        form.SeniorCitizen.value = '1';
        form.Partner.value = 'No';
        form.Dependents.value = 'No';
        form.PhoneService.value = 'Yes';
        form.MultipleLines.value = 'Yes';
        form.InternetService.value = 'Fiber optic';
        form.OnlineSecurity.value = 'No';
        form.OnlineBackup.value = 'No';
        form.DeviceProtection.value = 'No';
        form.TechSupport.value = 'No';
        form.StreamingTV.value = 'Yes';
        form.StreamingMovies.value = 'Yes';
        form.Contract.value = 'Month-to-month';
        form.PaperlessBilling.value = 'Yes';
        form.PaymentMethod.value = 'Electronic check';
        form.tenure.value = '3';
        form.MonthlyCharges.value = '98.50';
        form.TotalCharges.value = '295.50';
    } 
    else if (presetType === 'loyal') {
        // Loyal Safe Profile: 2-Year Contract, 58 months tenure, DSL, Tech Support, Automatic Bank Transfer
        form.gender.value = 'Male';
        form.SeniorCitizen.value = '0';
        form.Partner.value = 'Yes';
        form.Dependents.value = 'Yes';
        form.PhoneService.value = 'Yes';
        form.MultipleLines.value = 'Yes';
        form.InternetService.value = 'DSL';
        form.OnlineSecurity.value = 'Yes';
        form.OnlineBackup.value = 'Yes';
        form.DeviceProtection.value = 'Yes';
        form.TechSupport.value = 'Yes';
        form.StreamingTV.value = 'No';
        form.StreamingMovies.value = 'No';
        form.Contract.value = 'Two year';
        form.PaperlessBilling.value = 'No';
        form.PaymentMethod.value = 'Bank transfer (automatic)';
        form.tenure.value = '60';
        form.MonthlyCharges.value = '48.00';
        form.TotalCharges.value = '2880.00';
    } 
    else if (presetType === 'moderate') {
        // Moderate Profile: 1-Year Contract, 24 months tenure, moderate charges
        form.gender.value = 'Male';
        form.SeniorCitizen.value = '0';
        form.Partner.value = 'Yes';
        form.Dependents.value = 'No';
        form.PhoneService.value = 'Yes';
        form.MultipleLines.value = 'No';
        form.InternetService.value = 'DSL';
        form.OnlineSecurity.value = 'Yes';
        form.OnlineBackup.value = 'No';
        form.DeviceProtection.value = 'Yes';
        form.TechSupport.value = 'No';
        form.StreamingTV.value = 'Yes';
        form.StreamingMovies.value = 'No';
        form.Contract.value = 'One year';
        form.PaperlessBilling.value = 'Yes';
        form.PaymentMethod.value = 'Credit card (automatic)';
        form.tenure.value = '22';
        form.MonthlyCharges.value = '65.20';
        form.TotalCharges.value = '1434.40';
    }

    // Auto submit to instantly show results for smooth viva demonstration
    form.submit();
}
