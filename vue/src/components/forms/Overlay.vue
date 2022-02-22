<template>
  <div class="t-overlay">
    <LoadSpiner />
  </div>
</template>

<script>
import LoadSpiner from "@/components/forms/LoadSpiner";
export default {
  name: 'load-spiner',
  components: {
    LoadSpiner
  }
};
</script>
<style lang="scss" scoped>
.t-overlay {
  position: fixed;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  height: 100%;
  background-color: rgb(14 22 33 / 90%);
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
