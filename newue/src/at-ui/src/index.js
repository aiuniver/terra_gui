import Collapse from './components/collapse'
import CollapseItem from './components/collapse-item'
import Modal from './components/modal'
import locale from './locale'
import Dialog from './components/dialog'
import Message from './components/message'
import LoadingBar from './components/loading-bar'
import Notification from './components/notification'

const components = {
  Collapse,
  CollapseItem,
  Modal,
  Dialog,
  Message,
  LoadingBar,
  Notification
}

function install (Vue, opts = {}) {
  locale.use(opts.locale)
  locale.i18n(opts.i18n)

  for (const item in components) {
    if (components[item].name) {
      Vue.component(components[item].name, components[item])
    }
  }

  Vue.prototype.$Notify = Notification
  Vue.prototype.$Loading = LoadingBar
  Vue.prototype.$Modal = Dialog
  Vue.prototype.$Message = Message
}

/**
 * Global Install
 */
if (typeof window !== 'undefined' && window.Vue) {
  install(window.Vue)
}

const at = {
  install,
  locale: locale.use,
  i18n: locale.i18n,
  ...components
}

export default at;
