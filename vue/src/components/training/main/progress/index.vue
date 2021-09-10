<template>
  <div class="t-progress">
      <Timers />
      <Sysinfo />
  </div>
</template>

<script>
import Sysinfo from './Sysinfo.vue'
import Timers from './Timers.vue'

export default {
    name: 't-progress',
    components: {
        Sysinfo,
        Timers
    }
}
</script>

<style lang="scss" scoped>
.t-progress {
    font-size: 12px;
    line-height: 24px;
    color: #A7BED3;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}
</style>
