<template>
  <div class="fab" @click="$emit('click', $event)">
    <i class="fab__icon"></i>
  </div>
</template>

<script>
export default {
  name: "fab",
};
</script>

<style lang="scss" scoped>
.fab {
  width: 24px;
  height: 24px;
  background: #2b5278;
  border: 1px solid #65b9f4;
  box-sizing: border-box;
  border-radius: 4px;
  padding: 4px;
  cursor: pointer;
  &__icon {
    // background-size: 24px 24px;
    display: block;
    width: 14px;
    height: 14px;
    background-position: center;
    background-repeat: no-repeat;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggaWQ9IlZlY3RvciIgZD0iTTEzIDhIOFYxM0M4IDEzLjU1IDcuNTUgMTQgNyAxNEM2LjQ1IDE0IDYgMTMuNTUgNiAxM1Y4SDFDMC40NSA4IDAgNy41NSAwIDdDMCA2LjQ1IDAuNDUgNiAxIDZINlYxQzYgMC40NSA2LjQ1IDAgNyAwQzcuNTUgMCA4IDAuNDUgOCAxVjZIMTNDMTMuNTUgNiAxNCA2LjQ1IDE0IDdDMTQgNy41NSAxMy41NSA4IDEzIDhaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  }
}
</style>