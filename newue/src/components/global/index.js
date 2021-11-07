const requireComponent = require.context('@/components/global', true, /\.vue$/i, 'lazy');


console.log(requireComponent)
const components = []
requireComponent.keys().forEach(fileName => {
  console.log(requireComponent(fileName))
})

export { components }