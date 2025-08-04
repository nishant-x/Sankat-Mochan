import React, { useState, useEffect, useRef } from "react";
import { BsFillRecordCircleFill, BsRecordCircle } from "react-icons/bs";
import "./Chatbot.css";

const Chatbot = ({ userDetails }) => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(false); // Track speech enabled state
  const recognitionRef = useRef(null);
  const speechRef = useRef(null); // Track the current speech instance

  useEffect(() => {
    if ("webkitSpeechRecognition" in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + " ";
          }
        }
        setUserInput((prev) => prev + finalTranscript.trim());
      };

      recognitionRef.current = recognition;
    } else {
      alert("Speech recognition is not supported in this browser.");
    }
  }, []);

  const startRecording = () => {
    if (recognitionRef.current) {
      setIsRecording(true);
      recognitionRef.current.start();
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      setIsRecording(false);
      recognitionRef.current.stop();
    }
  };

  const textToSpeech = (text, lang = "en-US") => {
    if ("speechSynthesis" in window) {
      // Stop any ongoing speech before starting new one
      if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang; // Set the language based on input
      speechSynthesis.speak(utterance);
      speechRef.current = utterance; // Store current utterance
    } else {
      console.error("Text-to-Speech is not supported in this browser.");
    }
  };

  const sendMessage = async () => {
    if (userInput.trim() === "" || loading) return;

    const newMessages = [...messages, { sender: "user", text: userInput }];
    setMessages(newMessages);
    setUserInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8080/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userInput }),
      });

      if (!response.ok) {
        throw new Error(`Server Error: ${response.statusText}`);
      }

      const data = await response.json();
      const botResponse = data.response || "I'm here to help!";

      // Check if the response is in Hindi, change language for text-to-speech
      const isHindi = /[\u0900-\u097F]/.test(userInput); // Check if input contains Hindi characters
      const speechLang = isHindi ? "hi-IN" : "en-US"; // Use Hindi language for text-to-speech if detected

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: botResponse },
      ]);

      // Only start text-to-speech if isSpeechEnabled is true
      if (isSpeechEnabled) {
        textToSpeech(botResponse, speechLang);
      }
    } catch (error) {
      console.error("Error communicating with chatbot:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Sorry, I couldn't process that. Try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Toggle speech enabled state
  const toggleSpeech = () => {
    setIsSpeechEnabled((prev) => !prev);
  };

  return (
    <div className="chatbot_container">
      <div className="chatbot_messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat_message ${
              message.sender === "bot" ? "bot_message" : "user_message"
            }`}
          >
            {message.text}
          </div>
        ))}
        {loading && <div className="bot_message">Thinking...</div>}
      </div>

      <div className="chatbot_input">
        <input
          type="text"
          placeholder="Type a message..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          disabled={loading}
        />
        <button onClick={isRecording ? stopRecording : startRecording}>
          {isRecording ? <BsFillRecordCircleFill /> : <BsRecordCircle />}
        </button>
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>

        {/* Button to toggle text-to-speech */}
        <button onClick={toggleSpeech}>
          {isSpeechEnabled ? "Stop Sound" : "Start Sound"}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
