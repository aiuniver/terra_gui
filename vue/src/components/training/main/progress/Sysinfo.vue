<template>
  <div class="t-sysinfo">
    <div class="t-sysinfo__label">Информация об устройстве</div>
    <div class="t-sysinfo__grid">
      <div class="t-sysinfo__grid-item">GPU</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(gpu.gpu_utilization) }]">
        <p class="t-sysinfo__gpu-name">NVIDIA GeForce GTX 1060 6 GB</p>
        <p>{{ `${gpu.gpu_utilization || ''} (${gpu.gpu_memory_used || ''} / ${gpu.gpu_memory_total || ''})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: gpu.gpu_utilization }"></div>
        </div>
      </div>
      <div class="t-sysinfo__grid-item">RAM</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(ram.ram_utilization) }]">
        <p>{{ `${ram.ram_utilization || ''} (${ram.ram_memory_used || ''} / ${ram.ram_memory_total || ''})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: ram.ram_utilization }"></div>
        </div>
      </div>
      <div class="t-sysinfo__grid-item">Disk</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(disk.disk_utilization) }]">
        <p>{{ `${disk.disk_utilization || ''} (${disk.disk_memory_used || ''} / ${disk.disk_memory_total || ''})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: disk.disk_utilization }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: '',
  props: {
    data: Object,
  },
  computed: {
    disk() {
      return this.data?.Disk || {};
    },
    gpu() {
      return this.data?.GPU || {};
    },
    ram() {
      return this.data?.RAM || {};
    },
  },
  methods: {
    isWarning(value) {
      const int = value?.trim()?.replace('%', '');
      return +int > 50;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-sysinfo {
  display: flex;
  flex-direction: column;
  color: #a7bed3;
  gap: 30px;
  font-size: 14px;
  p {
    color: #fff;
  }
  &__gpu-name {
    color: #fff !important;
  }
  &__grid {
    display: grid;
    row-gap: 17px;
    column-gap: 30px;
    grid-template: repeat(3, max-content) / min-content 290px;
    &-item {
      &.warning p {
        color: #ca5035;
      }
      &.warning div div {
        background: #ca5035;
      }
    }
  }

  &__progress-bar {
    height: 5px;
    width: 100%;
    background: #242f3d;
    border-radius: 2.5px;
    margin-top: 5px;
    &--fill {
      background: #65ca35;
      border-radius: 2.5px;
      height: 100%;
    }
  }
}
</style>