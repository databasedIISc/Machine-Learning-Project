// this redirects the user from intro page to upload page
function upload() {
    window.location.href = "/upload";
}

// this redirects the user from homepage to intro page
function intro() {
    window.location.href = "/introduction";
}

// this redirects the user from upload page to main page where tool starts
function reachmain() {
    window.open("/main_page", "_blank");
}

function show_hist() {
    window.open("/histogram", "_blank");
}