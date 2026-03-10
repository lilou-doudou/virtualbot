export default function Chat() {
    return (
        <div className="p-m">
        <h3>Chat bot</h3>
<div className="p-mb-3">
 <input
     id="chat-input"
     type="text"
     className="p-inputtext p-component"
     style={{ width: "80%" }} // Définit la largeur à 100% du conteneur
     placeholder="Type your message..."
     aria-label="Message"
 />
    <button
        className="p-button p-component p-button-outlined"
        type="button"
        aria-label="Send message"
    >
        <span className="p-button-icon pi pi-check" />
        <span className="p-button-label">Send</span>
    </button>
</div>

        </div>
    );
    }