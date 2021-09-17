<template>
  <div class="t-progress">
    <div class="t-progress__item t-progress__item--timers">
      <Timers />
    </div>
    <div class="t-progress__item t-progress__item--info">
      <Sysinfo :usage="usage" />
    </div>
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
    usage() {
      return this.$store.getters['trainings/getUsage'];
    },
  },
  mounted() {
    console.log(this.data);
  },
};
</script>

<style lang="scss" scoped>
.t-progress {
  padding: 10px 0;
  font-size: 12px;
  line-height: 24px;
  color: #a7bed3;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 5px;
  &__item {

    &--timers {
    }
    &--info {
      width: 40%;
    }
  }
}
</style>
