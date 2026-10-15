const menuButton = document.querySelector('.menu-toggle');
const nav = document.querySelector('#main-nav');
if (menuButton && nav) {
  menuButton.addEventListener('click', () => {
    const expanded = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!expanded));
    nav.classList.toggle('open', !expanded);
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    nav.classList.remove('open');
    menuButton.setAttribute('aria-expanded', 'false');
  }));
}

async function copyText(text, button, defaultLabel) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copiado ✓';
  } catch (_) {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'absolute';
    area.style.left = '-9999px';
    document.body.appendChild(area);
    area.select();
    const copied = document.execCommand('copy');
    area.remove();
    button.textContent = copied ? 'Copiado ✓' : 'Selecione e copie';
  }
  window.setTimeout(() => { button.textContent = defaultLabel; }, 1800);
}

const responseButton = document.querySelector('#copy-response');
const responseCode = document.querySelector('#response-code');
if (responseButton && responseCode) {
  responseButton.addEventListener('click', () => copyText(responseCode.textContent, responseButton, 'Copiar exemplo ⧉'));
}
document.querySelectorAll('.copy-command').forEach(button => {
  button.addEventListener('click', () => copyText(button.dataset.copy || '', button, 'Copiar'));
});
