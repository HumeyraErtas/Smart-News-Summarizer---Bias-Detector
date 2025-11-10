function initializeBiasChart(biasScore) {
    const ctx = document.getElementById('biasChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Bias Skoru'],
                datasets: [
                    {
                        label: 'Bias',
                        data: [biasScore],
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }
}
