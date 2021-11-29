const proxy = {
  bondrogeen: 'http://192.168.1.47:8099/',
  MacBookPro: 'http://localhost:8001/',
  zabastx: 'http://174.138.5.111/',
};
module.exports = {
  productionSourceMap: false,
  devServer: {
    proxy: proxy[process.env.USERNAME] || 'http://localhost:8099/',
  },
  css:{
    loaderOptions:{
      scss:{
        prependData: `@import "~@/assets/scss/variables/default.scss";`
      }
    }
  }
};
