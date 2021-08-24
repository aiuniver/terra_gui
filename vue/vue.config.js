const proxy = {
  bondrogeen: 'http://192.168.1.47:8099/',
  // bondrogeen: 'http://idnq1zhruo.terra.neural-university.ru/',
  MacBookPro: 'http://localhost:8001/',
  zabastx: 'http://localhost:8000/'
}
module.exports = {
  productionSourceMap: false,
  devServer: {
    proxy: proxy[process.env.USERNAME] || 'http://localhost:8099/'
  }
}