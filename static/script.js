function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function hideEmptyState() {
    const empty = document.getElementById('emptyState');
    if (empty) empty.remove();
}

function addMessage(role, text, isThinking = false) {
    const messages = document.getElementById('messages');

    const wrap    = document.createElement('div');
    wrap.classList.add('message', role);
    if (isThinking) wrap.classList.add('thinking');

    const roleEl  = document.createElement('div');
    roleEl.classList.add('message-role');
    roleEl.textContent = role === 'user' ? 'You' : 'Model';

    const content = document.createElement('div');
    content.classList.add('message-content');

    if (isThinking) {
        content.innerHTML = 'Thinking<span class="thinking-dots"></span>';
    } else {
        content.textContent = text;
    }

    wrap.appendChild(roleEl);
    wrap.appendChild(content);
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;

    return { wrap, content };
}

async function sendMessage() {
    const input    = document.getElementById('questionInput');
    const sendBtn  = document.getElementById('sendBtn');
    const question = input.value.trim();

    if (!question) return;

    hideEmptyState();
    addMessage('user', question);

    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    const { wrap: thinkWrap, content: thinkContent } = addMessage('bot', '', true);

    try {
        const response = await fetch('/chat', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ question })
        });

        if (!response.ok) {
            thinkContent.textContent = 'Something went wrong.';
            thinkWrap.classList.remove('thinking');
            return;
        }

        const data = await response.json();
        thinkContent.textContent = data.answer;
        thinkWrap.classList.remove('thinking');

    } catch (err) {
        thinkContent.textContent = 'Could not reach the model.';
        thinkWrap.classList.remove('thinking');
        console.error(err);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function handleKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}