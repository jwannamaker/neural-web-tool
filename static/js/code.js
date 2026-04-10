async function createNetwork() {
    const response = await fetch('/create_network', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            layer_sizes: [3, 5, 2]
        })
    });
    const result = await response.json();
    console.log(result);
}