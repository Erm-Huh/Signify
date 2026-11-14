/*
 * view.js: tiny template runtime for the Signify Rev 3 pages (no build step, no dependencies).
 *
 * Each page has:
 *   <template id="view"> ... </template>   the markup, exactly as designed
 *   <script> class Component extends View { renderVals() { ... } } </script>
 *
 * Template syntax
 *   {{path}}                     text or attribute value from renderVals() (dotted lookup, e.g. {{item.label}})
 *   attr="{{path}}"              whole-value binding: passes the raw value (number, boolean, function)
 *   onClick / onChange / onSubmit="{{fn}}"   event handlers (onChange on text inputs fires on every keystroke)
 *   <sc-if value="{{cond}}">     render children only when cond is truthy
 *   <sc-for list="{{items}}" as="item">   repeat children; {{$index}} is available
 *
 * Component API (React-like): this.state, this.setState(partial), this.props,
 * componentDidMount(), componentWillUnmount(), renderVals().
 * Re-renders patch the existing DOM in place, so focus, scroll and CSS transitions are kept.
 */
(function () {
  const HOLE = /\{\{\s*([^}]+?)\s*\}\}/g;
  const WHOLE = /^\{\{\s*([^}]+?)\s*\}\}$/;

  function lookup(path, scope) {
    path = path.trim();
    if (path === 'true') return true;
    if (path === 'false') return false;
    if (/^-?\d+(\.\d+)?$/.test(path)) return Number(path);
    if (/^'.*'$|^".*"$/.test(path)) return path.slice(1, -1);
    return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), scope);
  }
  const interp = (str, scope) => str.replace(HOLE, (_, p) => { const v = lookup(p, scope); return v == null ? '' : String(v); });

  const EVENTS = { onclick: 'click', onchange: 'change', onsubmit: 'submit', oninput: 'input', onkeydown: 'keydown', onfocus: 'focus', onblur: 'blur' };
  const PROPS = { value: 'value', checked: 'checked', disabled: 'disabled', selected: 'selected' };

  // Build a fresh DOM tree (detached) from the template for the current values.
  function build(node, scope, out) {
    if (node.nodeType === 3) { out.appendChild(document.createTextNode(interp(node.nodeValue, scope))); return; }
    if (node.nodeType !== 1) return;
    const tag = node.tagName.toLowerCase();
    if (tag === 'sc-if') {
      const m = (node.getAttribute('value') || '').match(WHOLE);
      if (m && lookup(m[1], scope)) node.childNodes.forEach((c) => build(c, scope, out));
      return;
    }
    if (tag === 'sc-for') {
      const m = (node.getAttribute('list') || '').match(WHOLE);
      const list = (m && lookup(m[1], scope)) || [];
      const as = node.getAttribute('as') || 'item';
      list.forEach((item, i) => {
        const inner = Object.assign(Object.create(scope), { [as]: item, $index: i });
        node.childNodes.forEach((c) => build(c, inner, out));
      });
      return;
    }
    const el = document.createElement(tag);
    el.__on = {}; el.__props = {};
    for (const a of node.attributes) {
      const name = a.name.toLowerCase();
      const whole = a.value.match(WHOLE);
      if (EVENTS[name]) { const fn = whole && lookup(whole[1], scope); if (typeof fn === 'function') el.__on[EVENTS[name]] = fn; continue; }
      if (name.startsWith('hint-')) continue;
      if (PROPS[name] && whole) { el.__props[PROPS[name]] = lookup(whole[1], scope); continue; }
      if (whole) {
        const v = lookup(whole[1], scope);
        if (v === false || v == null) continue;
        el.setAttribute(name, v === true ? '' : String(v));
      } else el.setAttribute(name, interp(a.value, scope));
    }
    // Text inputs: React-style onChange fires on every keystroke.
    if (el.__on.change && (tag === 'textarea' || (tag === 'input' && !/^(checkbox|radio|range|file)$/.test(el.getAttribute('type') || '')))) {
      el.__on.input = el.__on.change; delete el.__on.change;
    }
    if (tag === 'input' && (el.getAttribute('type') || '') === 'range' && el.__on.change) el.__on.input = el.__on.change;
    node.childNodes.forEach((c) => build(c, scope, el));
    out.appendChild(el);
  }

  function wire(el) {
    el.__bound = el.__bound || {};
    for (const type of Object.keys(el.__on || {})) {
      if (!el.__bound[type]) { el.__bound[type] = true; el.addEventListener(type, (e) => { const fn = el.__on && el.__on[type]; if (fn) fn(e); }); }
    }
    for (const [k, v] of Object.entries(el.__props || {})) { if (el[k] !== v && !(k === 'value' && document.activeElement === el && String(el.value) === String(v))) el[k] = v == null ? '' : v; }
  }

  // Patch `live` children to match `next` children, reusing nodes where tags match.
  function patchChildren(live, next) {
    const a = Array.from(live.childNodes), b = Array.from(next.childNodes);
    b.forEach((nb, i) => {
      const la = a[i];
      if (!la) { live.appendChild(nb); wireTree(nb); return; }
      if (la.nodeType !== nb.nodeType || (la.nodeType === 1 && la.tagName !== nb.tagName)) { live.replaceChild(nb, la); wireTree(nb); return; }
      if (la.nodeType === 3) { if (la.nodeValue !== nb.nodeValue) la.nodeValue = nb.nodeValue; return; }
      for (const at of Array.from(la.attributes)) if (!nb.hasAttribute(at.name)) la.removeAttribute(at.name);
      for (const at of Array.from(nb.attributes)) if (la.getAttribute(at.name) !== at.value) la.setAttribute(at.name, at.value);
      la.__on = nb.__on; la.__props = nb.__props;
      wire(la);
      patchChildren(la, nb);
    });
    for (let i = a.length - 1; i >= b.length; i--) live.removeChild(a[i]);
  }
  function wireTree(n) { if (n.nodeType !== 1) return; wire(n); n.querySelectorAll('*').forEach(wire); }

  class View {
    constructor(props) { this.props = props || {}; this.state = {}; this._queued = false; }
    setState(partial) {
      Object.assign(this.state, typeof partial === 'function' ? partial(this.state) : partial);
      if (!this._queued) { this._queued = true; queueMicrotask(() => { this._queued = false; this._render(); }); }
    }
    forceUpdate() { this._render(); }
    renderVals() { return {}; }
    _render() {
      const frag = document.createElement('div');
      this._tpl.content.childNodes.forEach((c) => build(c, this.renderVals(), frag));
      patchChildren(this._root, frag);
    }
  }

  window.View = View;
  window.DCLogic = View; // alias: the design canvas calls the base class DCLogic

  document.addEventListener('DOMContentLoaded', () => {
    const tpl = document.getElementById('view');
    const root = document.getElementById('app');
    if (!tpl || !root || typeof window.Component !== 'function') return;
    const c = new window.Component({});
    c._tpl = tpl; c._root = root;
    c._render();
    if (c.componentDidMount) c.componentDidMount();
    window.addEventListener('pagehide', () => c.componentWillUnmount && c.componentWillUnmount());
  });
})();
