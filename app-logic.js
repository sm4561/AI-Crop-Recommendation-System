document.addEventListener('DOMContentLoaded', () => {
    // --- CACHE DOM ELEMENTS ---
    const form = document.getElementById('crop-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    const dashboard = document.getElementById('recommendation-dashboard');
    const cropSelector = document.getElementById('crop-selector');
    const getCropInfoButton = document.getElementById('get-crop-info');
    const cropInfoDisplay = document.getElementById('crop-info-display');
    const phSlider = document.getElementById('ph');
    const phValue = document.getElementById('ph-value');
    const humiditySlider = document.getElementById('humidity');
    const humidityValue = document.getElementById('humidity-value');
    const weatherButton = document.getElementById('get-location-weather');
    const cropSearchInput = document.getElementById('crop-search-input');
    let suitabilityChart;
    let allCropsList = [];

    // --- CROP DATA ---
    const cropData = {
        'rice': { icon: 'fa-seedling', yieldPerAcre: 24, marketPrice: 2200 }, 
        'maize': { icon: 'fa-fan', yieldPerAcre: 20, marketPrice: 2000 },
        'chickpea': { icon: 'fa-leaf', yieldPerAcre: 8, marketPrice: 4800 }, 
        'kidneybeans': { icon: 'fa-seedling', yieldPerAcre: 10, marketPrice: 6000 },
        'pigeonpeas': { icon: 'fa-seedling', yieldPerAcre: 6, marketPrice: 6500 }, 
        'mothbeans': { icon: 'fa-seedling', yieldPerAcre: 4, marketPrice: 7000 },
        'mungbean': { icon: 'fa-seedling', yieldPerAcre: 5, marketPrice: 7500 }, 
        'blackgram': { icon: 'fa-seedling', yieldPerAcre: 5, marketPrice: 7200 },
        'lentil': { icon: 'fa-seedling', yieldPerAcre: 7, marketPrice: 6000 }, 
        'pomegranate': { icon: 'fa-apple-whole', yieldPerAcre: 40, marketPrice: 10000 },
        'banana': { icon: 'fa-bacterium', yieldPerAcre: 200, marketPrice: 1500 }, 
        'mango': { icon: 'fa-lemon', yieldPerAcre: 40, marketPrice: 3000 },
        'grapes': { icon: 'fa-wine-glass', yieldPerAcre: 80, marketPrice: 4000 }, 
        'watermelon': { icon: 'fa-melon-slice', yieldPerAcre: 100, marketPrice: 1000 },
        'muskmelon': { icon: 'fa-melon-slice', yieldPerAcre: 60, marketPrice: 1200 }, 
        'apple': { icon: 'fa-apple-whole', yieldPerAcre: 80, marketPrice: 8000 },
        'orange': { icon: 'fa-apple-whole', yieldPerAcre: 100, marketPrice: 3500 }, 
        'papaya': { icon: 'fa-melon-slice', yieldPerAcre: 150, marketPrice: 800 },
        'coconut': { icon: 'fa-tree', yieldPerAcre: 50, marketPrice: 4000 }, 
        'cotton': { icon: 'fa-fan', yieldPerAcre: 10, marketPrice: 6000 },
        'jute': { icon: 'fa-seedling', yieldPerAcre: 15, marketPrice: 4500 }, 
        'coffee': { icon: 'fa-mug-hot', yieldPerAcre: 4, marketPrice: 20000 },
    };

    // --- SLIDER COLOR LOGIC ---
    const updateSliderBackground = (slider) => {
        const value = parseFloat(slider.value);
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const percentage = (value - min) / (max - min) * 100;
        slider.style.background = `linear-gradient(to right, #4f46e5 ${percentage}%, #d1d5db ${percentage}%)`;
    };

    // --- INITIALIZATION & HELPER FUNCTIONS ---
    const updateCropDropdown = (crops) => {
        cropSelector.innerHTML = '';
        if (crops.length === 0) {
            const option = document.createElement('option');
            option.textContent = 'No crops found';
            option.disabled = true;
            cropSelector.appendChild(option);
        } else {
            crops.forEach(crop => {
                const option = document.createElement('option');
                option.value = crop;
                option.textContent = crop.charAt(0).toUpperCase() + crop.slice(1);
                cropSelector.appendChild(option);
            });
        }
    };

    const populateCropSelector = async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/get_crop_names');
            if (!response.ok) throw new Error(`Network response was not ok`);
            const data = await response.json();
            allCropsList = data.crop_names;
            updateCropDropdown(allCropsList);
            if (window.setLanguage) window.setLanguage(localStorage.getItem('language') || 'en');
        } catch (error) {
            console.error("Failed to fetch crop names:", error);
        }
    };
    
    // --- EVENT LISTENERS ---
    cropSearchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const filteredCrops = allCropsList.filter(crop => crop.toLowerCase().includes(searchTerm));
        updateCropDropdown(filteredCrops);
    });

    phSlider.addEventListener('input', () => { 
        phValue.textContent = parseFloat(phSlider.value).toFixed(1); 
        updateSliderBackground(phSlider);
    });
    humiditySlider.addEventListener('input', () => { 
        humidityValue.textContent = `${humiditySlider.value}%`; 
        updateSliderBackground(humiditySlider);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        cropInfoDisplay.classList.add('hidden');
        dashboard.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        const inputData = {
            N: parseFloat(document.getElementById('nitrogen').value),
            P: parseFloat(document.getElementById('phosphorus').value),
            K: parseFloat(document.getElementById('potassium').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(humiditySlider.value),
            ph: parseFloat(phSlider.value),
            rainfall: parseFloat(document.getElementById('rainfall').value),
        };
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const recommendations = await response.json();
            updateDashboard(recommendations);
        } catch (error) {
            console.error("API call failed:", error);
            alert("Failed to get recommendations. Ensure the backend is running.");
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    getCropInfoButton.addEventListener('click', async () => {
        const selectedCrop = cropSelector.value;
        if (!selectedCrop || cropSelector.options[cropSelector.selectedIndex].disabled) {
            return alert("Please select a valid crop from the list.");
        }
        try {
            dashboard.classList.add('hidden');
            loadingSpinner.classList.remove('hidden');
            const response = await fetch('http://127.0.0.1:5000/get_crop_info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop_name: selectedCrop })
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            displayCropInfo(data);
        } catch (error) {
            console.error("Failed to fetch crop info:", error);
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    weatherButton.addEventListener('click', () => {
        if (!navigator.geolocation) return alert("Geolocation is not supported.");
        const weatherButtonText = document.getElementById('weather-button-text');
        const weatherSpinner = document.getElementById('weather-spinner');
        weatherButtonText.textContent = 'Fetching...';
        if (window.setLanguage) window.setLanguage(localStorage.getItem('language') || 'en');
        weatherSpinner.classList.remove('hidden');
        weatherButton.disabled = true;

        navigator.geolocation.getCurrentPosition(async (position) => {
            const { latitude: lat, longitude: lon } = position.coords;
            const apiKey = 'e949eb1087ff76cca1464777fd54c562';
            const apiUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;
            try {
                const response = await fetch(apiUrl);
                if (!response.ok) throw new Error('Could not fetch weather data.');
                const weatherData = await response.json();
                document.getElementById('temperature').value = weatherData.main.temp.toFixed(1);
                humiditySlider.value = weatherData.main.humidity;
                humidityValue.textContent = `${weatherData.main.humidity}%`;
                updateSliderBackground(humiditySlider);
            } catch (error) {
                console.error("Weather API Error:", error);
                alert(error.message);
            } finally {
                weatherButtonText.textContent = 'Use My Location';
                if (window.setLanguage) window.setLanguage(localStorage.getItem('language') || 'en');
                weatherSpinner.classList.add('hidden');
                weatherButton.disabled = false;
            }
        }, (error) => {
            console.error("Geolocation Error:", error);
            alert("Could not get your location. Please grant permission.");
            weatherButtonText.textContent = 'Use My Location';
            if (window.setLanguage) window.setLanguage(localStorage.getItem('language') || 'en');
            weatherSpinner.classList.add('hidden');
            weatherButton.disabled = false;
        });
    });

    document.getElementById('calculate-profit').addEventListener('click', () => {
        const landSize = parseFloat(document.getElementById('land-size').value);
        const marketPrice = parseFloat(document.getElementById('market-price').value);
        const topCropName = document.getElementById('top-crop-name').textContent.toLowerCase();
        if (isNaN(landSize) || isNaN(marketPrice) || landSize <= 0 || marketPrice <= 0) {
            return alert("Please enter valid land size and market price.");
        }
        const yieldPerAcre = cropData[topCropName]?.yieldPerAcre || 15;
        const totalYield = landSize * yieldPerAcre;
        const totalProfit = totalYield * marketPrice;
        document.getElementById('expected-yield').textContent = totalYield.toFixed(2);
        document.getElementById('estimated-profit').textContent = totalProfit.toLocaleString('en-IN');
        document.getElementById('profit-result').classList.remove('hidden');
    });

    // --- DISPLAY FUNCTIONS ---
    function displayCropInfo(data) {
        const cropName = data.crop_name.charAt(0).toUpperCase() + data.crop_name.slice(1);
        const detailsContainer = document.getElementById('crop-info-details');
        const guide = data.guide;
        document.getElementById('crop-info-name').textContent = `${cropName} Required Content`;

        const createLabeledItem = (label, value, iconClass) => `
            <div class="bg-gray-100 dark:bg-gray-700/50 p-3 rounded-lg flex items-center space-x-3">
                <i class="fas ${iconClass} text-indigo-500 text-xl w-6 text-center"></i>
                <div>
                    <p class="text-sm font-medium text-gray-500 dark:text-gray-400">${label}</p>
                    <p class="text-lg font-bold text-gray-800 dark:text-white">${value}</p>
                </div>
            </div>`;

        detailsContainer.innerHTML = `
            ${createLabeledItem("Nitrogen (N)", guide.idealNitrogen, 'fa-seedling')}
            ${createLabeledItem("Phosphorus (P)", guide.idealPhosphorus, 'fa-leaf')}
            ${createLabeledItem("Potassium (K)", guide.idealPotassium, 'fa-fan')}
            ${createLabeledItem("Soil pH", guide.idealPh, 'fa-flask')}
            ${createLabeledItem("Temperature", `~${guide.idealTemp}°C`, 'fa-thermometer-half')}
            ${createLabeledItem("Humidity", `~${guide.idealHumidity}%`, 'fa-water')}
            ${createLabeledItem("Rainfall", `~${guide.idealRainfall}mm`, 'fa-cloud-showers-heavy')}
        `;
        cropInfoDisplay.classList.remove('hidden');
    }
    
    const updateTipsDisplay = (tips) => {
        const analysisList = document.getElementById('analysis-details');
        analysisList.innerHTML = ''; 

        const statusMap = {
            low: { icon: 'fa-arrow-down', color: 'text-red-500' },
            high: { icon: 'fa-arrow-up', color: 'text-red-500' },
            good: { icon: 'fa-check-circle', color: 'text-green-500' }
        };

        if (tips.analysis && tips.analysis.length > 0) {
            tips.analysis.forEach(item => {
                const statusInfo = statusMap[item.status];
                const li = document.createElement('li');
                li.className = 'flex justify-between items-center';
                li.innerHTML = `
                    <div class="flex items-center">
                        <i class="fas ${statusInfo.icon} ${statusInfo.color} w-5 text-center mr-3"></i>
                        <span class="font-semibold">${item.parameter}</span>
                    </div>
                    <div class="text-right">
                        <span>Your Value: <span class="font-bold">${item.user_value}</span></span>
                        <span class="ml-4">Ideal: <span class="font-bold text-green-600 dark:text-green-400">${item.ideal_value}</span></span>
                    </div>
                `;
                analysisList.appendChild(li);
            });
        } else {
             analysisList.innerHTML = '<p class="text-gray-600 dark:text-gray-400">No specific analysis available.</p>';
        }

        document.getElementById('fertilizer-tip').textContent = tips.fertilizer;
        document.getElementById('irrigation-tip').textContent = tips.irrigation;
    };

    function updateDashboard(data) {
        const topCrop = data.top_crop;
        const topCropName = topCrop.name.charAt(0).toUpperCase() + topCrop.name.slice(1);
        document.getElementById('top-crop-name').textContent = topCropName;
        document.getElementById('top-crop-suitability').textContent = `${topCrop.suitability}%`;
        document.getElementById('crop-icon').className = `fas ${cropData[topCrop.name.toLowerCase()]?.icon || 'fa-seedling'} text-7xl text-white`;
        
        const marketPriceInput = document.getElementById('market-price');
        const recommendedCropData = cropData[topCrop.name.toLowerCase()];
        if (recommendedCropData && recommendedCropData.marketPrice) {
            marketPriceInput.value = recommendedCropData.marketPrice;
        } else {
            marketPriceInput.value = 2000;
        }
        document.getElementById('profit-result').classList.add('hidden');

        const otherCropsList = document.getElementById('other-crops-list');
        otherCropsList.innerHTML = '';
        data.other_crops.forEach(crop => {
            const li = document.createElement('li');
            li.className = 'flex justify-between items-center text-sm py-1';
            li.innerHTML = `<span>${crop.name.charAt(0).toUpperCase() + crop.name.slice(1)}</span><span class="font-semibold">${crop.suitability}%</span>`;
            otherCropsList.appendChild(li);
        });
        updateTipsDisplay(data.tips);
        renderChart(data);
        dashboard.classList.remove('hidden');
    }

    function updateChartTheme() {
        if (!suitabilityChart) return;
        const isDarkMode = document.documentElement.classList.contains('dark');
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#E5E7EB' : '#374151';
        suitabilityChart.options.scales.y.ticks.color = textColor;
        suitabilityChart.options.scales.y.grid.color = gridColor;
        suitabilityChart.options.scales.x.ticks.color = textColor;
        suitabilityChart.options.scales.x.grid.color = gridColor;
        suitabilityChart.update();
    }
    const themeToggleButton = document.getElementById('theme-toggle');
    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => setTimeout(updateChartTheme, 50));
    }
    
    function renderChart(data) {
        const canvas = document.getElementById('suitability-chart');
        if (!canvas) return console.error('Suitability chart canvas not found!');
        const ctx = canvas.getContext('2d');
        const allCrops = [data.top_crop, ...data.other_crops];
        const labels = allCrops.map(c => c.name.charAt(0).toUpperCase() + c.name.slice(1));
        const values = allCrops.map(c => c.suitability);
        if (suitabilityChart) suitabilityChart.destroy();
        suitabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Suitability %',
                    data: values,
                    backgroundColor: 'rgba(34, 197, 94, 0.6)',
                    borderColor: 'rgba(22, 163, 74, 1)',
                    borderWidth: 1,
                    borderRadius: 5,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 100 }, x: {} },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: (context) => `Suitability: ${context.raw}%` } }
                }
            }
        });
        updateChartTheme();
    }
    
    function setupAccordion() {
        const accordionButtons = document.querySelectorAll('.accordion-button');
        accordionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const content = button.nextElementSibling;
                const icon = button.querySelector('.accordion-icon');
                
                document.querySelectorAll('.accordion-content').forEach(cont => {
                    if (cont !== content) {
                        cont.classList.add('hidden');
                    }
                });
                document.querySelectorAll('.accordion-icon').forEach(ic => {
                    if (ic !== icon) {
                        ic.classList.remove('rotate-180');
                    }
                });
                
                content.classList.toggle('hidden');
                icon.classList.toggle('rotate-180');
            });
        });
    }

    // --- NEW: WELCOME MODAL LOGIC ---
    function setupWelcomeModal() {
        const modal = document.getElementById('welcome-modal');
        const backdrop = document.getElementById('welcome-modal-backdrop');
        const closeButton = document.getElementById('close-welcome-modal');

        const closeModal = () => {
            modal.classList.add('hidden');
            backdrop.classList.add('hidden');
            // Remember that the user has seen the modal
            localStorage.setItem('hasSeenWelcomeModal', 'true');
        };

        closeButton.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);

        // Check if the user has visited before
        if (!localStorage.getItem('hasSeenWelcomeModal')) {
            modal.classList.remove('hidden');
            backdrop.classList.remove('hidden');
        }
    }


    // --- Run initial setup functions ---
    populateCropSelector();
    setupAccordion();
    updateSliderBackground(phSlider);
    updateSliderBackground(humiditySlider);
    setupWelcomeModal(); // Initialize the welcome modal
});








