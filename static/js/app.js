// static/js/app.js

// Wait for the HTML DOM to be fully loaded before running script
document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Element References ---
    const coinSelectButton = document.getElementById('coinSelectButton');
    const coinSelectList = document.getElementById('coinSelectList');
    const chartCanvas = document.getElementById('priceChart');
    const chartTitle = document.getElementById('chartTitle');
    const chartLoading = document.getElementById('chartLoading'); // Chart loading indicator
    const predictionCardEOD = document.getElementById('prediction-card-eod');
    const predictionCardNextOpen = document.getElementById('prediction-card-next-open');
    const predictionValueEOD = predictionCardEOD.querySelector('.prediction-value');
    const predictionValueNextOpen = predictionCardNextOpen.querySelector('.prediction-value');
    const disclaimerTextElement = document.getElementById('disclaimer-text'); // If you want to set disclaimer via JS

    let currentChartInstance = null; // Variable to hold the Chart.js instance
    let selectedCoinId = null; // Variable to hold the currently selected coin ID

    // --- Functions ---

    /**
     * Toggles loading state visuals for predictions and chart
     * @param {boolean} isLoading - True to show loading, false to hide
     */
    function setLoadingState(isLoading) {
        // Check if the chartLoading element exists before trying to use it
        const chartLoading = document.getElementById('chartLoading'); // Re-get or ensure it's accessible here
    
        if (isLoading) {
            // Show loading for predictions
            predictionCardEOD?.classList.add('loading');
            predictionCardNextOpen?.classList.add('loading');
            predictionValueEOD.textContent = '--';
            predictionValueNextOpen.textContent = '--';
    
            // Show loading for chart (with check)
            if (chartLoading) { // <-- ADD CHECK
                chartLoading.style.display = 'block';
            }
            if (currentChartInstance) {
                chartCanvas.style.opacity = '0.3'; // Use the corrected line here
            }
        } else {
            // Hide loading for predictions
            predictionCardEOD?.classList.remove('loading');
            predictionCardNextOpen?.classList.remove('loading');
    
            // Hide loading for chart (with check)
            if (chartLoading) { // <-- ADD CHECK
                chartLoading.style.display = 'none';
            }
            if (chartCanvas) { // Also good to check chartCanvas
                chartCanvas.style.opacity = '1';
            }
        }
    }

    /**
     * Updates the prediction display cards.
     * @param {object | null} predictions - Prediction data object or null on error.
     */
    function updatePredictions(predictions) {
        if (predictions && predictions.predicted_eod_close !== undefined) {
            // Format as currency (e.g., $12,345.67)
            const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
            predictionValueEOD.textContent = formatter.format(predictions.predicted_eod_close);
            predictionValueNextOpen.textContent = formatter.format(predictions.predicted_next_open);
        } else {
            predictionValueEOD.textContent = 'Error';
            predictionValueNextOpen.textContent = 'Error';
            console.error("Error fetching or displaying predictions:", predictions?.error || "Unknown error");
        }
    }

    /**
     * Fetches prediction data from the backend.
     * @param {string} coinId - The ID of the coin (e.g., 'bitcoin').
     */
    async function fetchPredictionData(coinId) {
        try {
            const response = await fetch(`/predict/${coinId}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const predictions = await response.json();
            updatePredictions(predictions); // Update prediction display
        } catch (error) {
            console.error('Error fetching prediction data:', error);
            updatePredictions(null); // Show error state
        }
    }

    /**
     * Updates the Chart.js chart with new data.
     * @param {Array} data - Array of {date, price} objects.
     * @param {string} coinName - Capitalized name of the coin for the title.
     */
    function updateChart(data, coinName) {
        if (!chartCanvas) return; // Exit if canvas element isn't found
        const ctx = chartCanvas.getContext('2d');

        // Prepare data for Chart.js
        const labels = data.map(item => item.date.split('T')[0]); // Extract YYYY-MM-DD
        const prices = data.map(item => item.price);

        // Update chart title
        chartTitle.textContent = `${coinName} Historical Price Chart (USD)`;

        // Destroy previous chart instance if it exists
        if (currentChartInstance) {
            currentChartInstance.destroy();
        }

        // Create new chart instance
        currentChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `${coinName} Price`,
                    data: prices,
                    borderColor: 'rgb(75, 192, 192)', // Teal color line
                    backgroundColor: 'rgba(75, 192, 192, 0.1)', // Slight fill under line
                    tension: 0.1, // Slight curve to the line
                    pointRadius: 1, // Smaller points
                    pointHoverRadius: 5,
                    fill: true // Enable fill
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // Allow chart to fill container height
                scales: {
                    x: {
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)', // Light ticks for dark theme
                             maxRotation: 0, // Prevent label rotation
                             autoSkip: true, // Skip some labels if too crowded
                             maxTicksLimit: 10 // Limit number of X-axis ticks shown
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)' // Light grid lines
                        }
                    },
                    y: {
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            // Format Y-axis ticks as currency
                            callback: function(value, index, values) {
                                return '$' + value.toLocaleString();
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false // Hide legend if only one dataset
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: { // Customize tooltip
                             label: function(context) {
                                 let label = context.dataset.label || '';
                                 if (label) { label += ': '; }
                                 if (context.parsed.y !== null) {
                                     label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                 }
                                 return label;
                             }
                        }
                    }
                },
                interaction: { // Optimize interaction modes
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

     /**
     * Fetches historical data for charting from the backend.
     * @param {string} coinId - The ID of the coin (e.g., 'bitcoin').
     */
     async function fetchHistoricalData(coinId) {
        try {
            const response = await fetch(`/historical_data/${coinId}`);
            if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
             if(data && data.length > 0){
                 updateChart(data, coinId.charAt(0).toUpperCase() + coinId.slice(1)); // Update the chart
             } else {
                  updateChart([], coinId.charAt(0).toUpperCase() + coinId.slice(1)); // Clear chart if no data
                  console.warn("Received empty historical data for", coinId);
             }
        } catch (error) {
            console.error('Error fetching historical data:', error);
            updateChart([], coinId.charAt(0).toUpperCase() + coinId.slice(1)); // Clear chart on error
             // Optionally display an error message to the user near the chart
        } finally {
             // This is now handled by setLoadingState(false) after both fetches complete
        }
    }

    /**
     * Fetches both historical and prediction data for the selected coin.
     * @param {string} coinId - The ID of the coin.
     */
    async function loadCoinData(coinId) {
        if (!coinId) return;

        selectedCoinId = coinId; // Store selected coin
        console.log(`Loading data for: ${coinId}`);
        setLoadingState(true); // Show loading indicators

        // Update the button text
        coinSelectButton.textContent = coinId.charAt(0).toUpperCase() + coinId.slice(1);

        // Fetch data in parallel
        try {
            await Promise.all([
                fetchHistoricalData(coinId),
                fetchPredictionData(coinId)
            ]);
        } catch (error) {
             console.error(`Error loading data for ${coinId}:`, error);
             // Error state is handled within individual fetch functions
        } finally {
            setLoadingState(false); // Hide loading indicators regardless of success/error
        }
    }

    // --- Event Listeners ---
    if (coinSelectList) {
        coinSelectList.addEventListener('click', (event) => {
            // Check if the clicked element is a dropdown item with a coinid
            if (event.target && event.target.matches('.dropdown-item') && event.target.dataset.coinid) {
                event.preventDefault(); // Prevent default anchor behavior
                const coinId = event.target.dataset.coinid;
                loadCoinData(coinId);
            }
        });
    } else {
         console.error("Coin select list element not found.");
    }

    // --- Initial Load ---
    // Find the first available coin from the dropdown list rendered by Flask
    const firstCoinItem = coinSelectList?.querySelector('.dropdown-item[data-coinid]');
    if (firstCoinItem) {
        const initialCoinId = firstCoinItem.dataset.coinid;
        loadCoinData(initialCoinId); // Load data for the first coin on page load
    } else {
         console.warn("No initial coin found to load.");
          // Handle state where no models might be loaded or list is empty
          coinSelectButton.textContent = "N/A";
          predictionValueEOD.textContent = 'N/A';
          predictionValueNextOpen.textContent = 'N/A';
          chartTitle.textContent = "No Data Available";
    }

    // Disclaimer Placeholder Handling (Optional)
    // If you want to set the disclaimer dynamically, you could fetch it or set it here.
    // For now, we assume the placeholder in index.html is sufficient or you'll replace it manually.
    // Example: disclaimerTextElement.textContent = "Your fetched/configured disclaimer text";

}); // End DOMContentLoaded