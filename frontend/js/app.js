// FastAPI backend
const API_URL = "http://127.0.0.1:8000/chat";


// Get HTML elements
const chat = document.getElementById("chat");

const questionInput =
    document.getElementById("question");

const sendButton =
    document.getElementById("sendButton");


// Send question when button is clicked
sendButton.addEventListener(
    "click",
    sendQuestion
);


// Send question when Enter is pressed
questionInput.addEventListener(
    "keydown",
    function (event) {

        if (event.key === "Enter") {

            sendQuestion();

        }

    }
);


// Main function
async function sendQuestion() {

    const question =
        questionInput.value.trim();


    // Don't send empty question
    if (!question) {

        return;

    }


    // Show user message
    addMessage(
        "You",
        question,
        "user"
    );


    // Clear input
    questionInput.value = "";


    // Disable button
    sendButton.disabled = true;


    // Show loading message
    const loadingMessage =
        addMessage(
            "Bot",
            "Thinking...",
            "bot",
            true
        );


    try {

        // Send request to FastAPI
        const response =
            await fetch(API_URL, {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question: question
                })

            });


        // Convert response to JSON
        const data =
            await response.json();


        // Remove loading message
        loadingMessage.remove();


        // API error
        if (!response.ok) {

            addMessage(
                "Bot",
                "Request failed.",
                "bot"
            );

            console.error(data);

            return;
        }


        // Show chatbot answer
        addMessage(
            "Bot",
            data.answer,
            "bot",
            false,
            data
        );


    } catch (error) {

        // Remove loading message
        loadingMessage.remove();


        // Connection error
        addMessage(
            "Bot",
            "Could not connect to the FastAPI server.",
            "bot"
        );


        console.error(
            "Connection error:",
            error
        );

    } finally {

        // Enable button
        sendButton.disabled = false;

        // Focus input
        questionInput.focus();

    }

}


// Add message to chat
function addMessage(
    sender,
    text,
    type,
    isLoading = false,
    data = null
) {

    const message =
        document.createElement("div");


    message.className =
        "message " + type;


    if (isLoading) {

        message.classList.add("loading");

    }


    // Message text
    message.innerHTML =
        "<strong>" +
        escapeHTML(sender) +
        ":</strong><br>" +
        escapeHTML(text);


    // Sources
    if (
        data &&
        data.sources &&
        data.sources.length > 0
    ) {

        let sourcesHTML =
            "<div class='sources'>" +
            "<strong>Sources:</strong><br>";


        data.sources.forEach(
            function (source) {

                sourcesHTML +=
                    "Section " +
                    escapeHTML(
                        String(
                            source.section_number
                        )
                    ) +
                    ": " +
                    escapeHTML(
                        source.heading
                    ) +
                    "<br>";

            }
        );


        sourcesHTML += "</div>";


        message.innerHTML +=
            sourcesHTML;
    }


    // Timing information
    if (data) {

        message.innerHTML +=

            "<div class='timing'>" +

            "Retrieval: " +

            Number(
                data.retrieval_time
            ).toFixed(4) +

            "s | Generation: " +

            Number(
                data.generation_time
            ).toFixed(4) +

            "s | Total: " +

            Number(
                data.total_time
            ).toFixed(4) +

            "s" +

            "</div>";
    }


    // Add to chat
    chat.appendChild(message);


    // Scroll to bottom
    chat.scrollTop =
        chat.scrollHeight;


    return message;
}


// Protect HTML from user input
function escapeHTML(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}