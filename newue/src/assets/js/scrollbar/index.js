import vuescroll from './vuescroll-native.min.js';

const settings = {
  ops: {
    vuescroll: {
      mode: 'native',
      sizeStrategy: 'percent',
      detectResize: false,
      locking: true,
    },
    bar: {
      showDelay: 500,
      onlyShowBarOnScroll: false,
      keepShow: true,
      background: '#4d6078',
      opacity: 1,
      hoverStyle: false,
      specifyBorderRadius: false,
      minSize: 0,
      size: '4px',
      disable: false
    },
    rail: {
      gutterOfEnds: '3px',
      gutterOfSide: '3px',
      background: '#01a99a',
      opacity: 0,
      size: '4px',
    }
  },
  name: 'scrollbar'
}

export { vuescroll, settings }