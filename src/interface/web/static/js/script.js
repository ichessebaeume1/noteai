$(document).ready(function() {
    $('#topicForm').submit(function(event) {
        event.preventDefault();
        const topic = $('#topicInput').val();
        $.post('/process_input', { topic: topic }, function(response) {
            $('#summary').text(response.message);
        });
    });
});
