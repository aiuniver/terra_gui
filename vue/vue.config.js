const proxy = {
  bondrogeen: 'http://192.168.1.47:8000/',
  // bondrogeen: 'http://idnq1zhruo.terra.neural-university.ru/',
  MacBookPro: 'http://localhost:8001/'
}
module.exports = {
  devServer: {
    proxy: proxy[process.env.USERNAME] || 'http://localhost:8000/'
  }
}