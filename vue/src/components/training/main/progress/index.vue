<template>
  <div class="t-progress">
    <Timers />
    <Sysinfo :data="data"/>
  </div>
</template>

<script>
import Sysinfo from './Sysinfo.vue';
import Timers from './Timers.vue';

export default {
  name: 't-progress',
  components: {
    Sysinfo,
    Timers,
  },
  computed: {
    data() {
      return this.$store.getters['trainings/getTest'].usage;
    },
  },
  mounted() {
    console.log(this.data);
  },
};
</script>

<style lang="scss" scoped>
.t-progress {
  font-size: 12px;
  line-height: 24px;
  color: #a7bed3;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
}
</style>
