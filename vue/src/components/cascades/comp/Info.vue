<template>
  <span class="info">
      <i @click="showTip=!showTip"></i>
      <p v-show="showTip">
          Этот параметр активируйте в случае когда необходимо
          применить постобработку выходных данных модели для
          последующих блоков или вывода окончательного результата.
      </p>
  </span>
</template>

<script>
export default {
    name: 'info',
    data: () => ({
        showTip: false
    })
}
</script>

<style lang="scss" scoped>
.info {
    position: relative;
    display: inline-block;
    i {
        display: inline-block;
        cursor: pointer;
        width: 20px;
        height: 20px;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTkgNUgxMVY3SDlWNVpNMTAgMTVDMTAuNTUgMTUgMTEgMTQuNTUgMTEgMTRWMTBDMTEgOS40NSAxMC41NSA5IDEwIDlDOS40NSA5IDkgOS40NSA5IDEwVjE0QzkgMTQuNTUgOS40NSAxNSAxMCAxNVpNMTAgMEM0LjQ4IDAgMCA0LjQ4IDAgMTBDMCAxNS41MiA0LjQ4IDIwIDEwIDIwQzE1LjUyIDIwIDIwIDE1LjUyIDIwIDEwQzIwIDQuNDggMTUuNTIgMCAxMCAwWk0xMCAxOEM1LjU5IDE4IDIgMTQuNDEgMiAxMEMyIDUuNTkgNS41OSAyIDEwIDJDMTQuNDEgMiAxOCA1LjU5IDE4IDEwQzE4IDE0LjQxIDE0LjQxIDE4IDEwIDE4WiIgZmlsbD0iIzY1QjlGNCIvPgo8L3N2Zz4K');
    }
    p {
        position: absolute;
        background: #2B5278;
        opacity: 0.9;
        border-radius: 4px;
        padding: 5px;
        top: 27px;
        right: 0;
        width: 200px;
        font-size: 8px;
        line-height: 11px;
    }
}
</style>