import 'es6-object-assign/auto';

const checkWidth = function (el, binding) {
  const mirror = document.querySelector(`.vue-input-autowidth-mirror-${el.dataset.uuid}`);
  const defaults = { maxWidth: 'none', minWidth: 'none', comfortZone: 0 };
  const options = Object.assign({}, defaults, binding.value);
  el.style.maxWidth = options.maxWidth;
  el.style.minWidth = options.minWidth;
  let val = el.value;
  if (!val) {
    val = el.placeholder || '';
  }
  while (mirror.childNodes.length) {
    mirror.removeChild(mirror.childNodes[0]);
  }
  mirror.appendChild(document.createTextNode(val));
  let newWidth = mirror.scrollWidth + options.comfortZone + 2;
  if (newWidth != el.scrollWidth) {
    el.style.width = `${newWidth}px`;
  }
};

export default {
  name: 'autowidth',
  bind: function (el) {
    if (el.tagName.toLocaleUpperCase() !== 'INPUT') {
      throw new Error('v-input-autowidth can only be used on input elements.');
    }
    el.dataset.uuid = Math.random().toString(36).slice(-5);
    el.style.boxSizing = 'content-box';
  },
  inserted: function (el, binding, vnode) {
    const hasVModel = vnode.data.directives.some(directive => directive.name === 'model');
    const styles = window.getComputedStyle(el);
    el.mirror = document.createElement('div');
    Object.assign(el.mirror.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      visibility: 'hidden',
      height: '0',
      overflow: 'hidden',
      whiteSpace: 'pre',
      fontSize: styles.fontSize,
      fontFamily: styles.fontFamily,
      fontWeight: styles.fontWeight,
      fontStyle: styles.fontStyle,
      letterSpacing: styles.letterSpacing,
      textTransform: styles.textTransform,
    });
    el.mirror.classList.add(`vue-input-autowidth-mirror-${el.dataset.uuid}`);
    el.mirror.setAttribute('aria-hidden', 'true');
    document.body.appendChild(el.mirror);
    checkWidth(el, binding);
    if (!hasVModel) {
      el.addEventListener('input', checkWidth.bind(null, el, binding));
    }
  },
  componentUpdated: function (el, binding) {
    checkWidth(el, binding);
  },
  unbind: function (el, binding) {
    document.body.removeChild(el.mirror);
    el.removeEventListener('input', checkWidth.bind(null, el, binding));
  },
};
