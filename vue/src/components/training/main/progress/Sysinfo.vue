<template>
  <div class="t-sysinfo">
    <div class="t-sysinfo__label">Информация об устройстве</div>
    <div class="t-sysinfo__grid">
      <div class="t-sysinfo__grid-item">GPU</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(gpu.gpu_utilization) }]">
        <p class="t-sysinfo__gpu-name">NVIDIA GeForce GTX 1060 6 GB</p>
        <p>{{ `${gpu.gpu_utilization || '0%'} (${gpu.gpu_memory_used || '0'} / ${gpu.gpu_memory_total || '0'})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: (gpu.gpu_utilization || 0) + '%' }"></div>
        </div>
      </div>
      <div class="t-sysinfo__grid-item">CPU</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(cpu.cpu_utilization) }]">
        <p>{{ `${cpu.cpu_utilization || '0%'} (${cpu.cpu_memory_used || '0'} / ${cpu.cpu_memory_total || '0'})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: (cpu.cpu_utilization || 0) + '%' }"></div>
        </div>
      </div>
      <div class="t-sysinfo__grid-item">RAM</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(ram.ram_utilization) }]">
        <p>{{ `${ram.ram_utilization + '% '  || '0%'} (${ram.ram_memory_used || '0'} / ${ram.ram_memory_total || '0'})` }}</p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: (ram.ram_utilization || 0) + '%' }"></div>
        </div>
      </div>
      <div class="t-sysinfo__grid-item">Disk</div>
      <div :class="['t-sysinfo__grid-item', { warning: isWarning(disk.disk_utilization) }]">
        <p>
          {{ `${disk.disk_utilization + '% ' || '0%'} (${disk.disk_memory_used || '0'} / ${disk.disk_memory_total || '0'})` }}
        </p>
        <div class="t-sysinfo__progress-bar">
          <div class="t-sysinfo__progress-bar--fill" :style="{ width: (disk.disk_utilization || 0) + '%' }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: '',
  props: {
    usage: Object,
  },
  computed: {
    disk() {
      return this.usage?.Disk || {};
    },
    gpu() {
      return this.usage?.GPU || {};
    },
    cpu() {
      return this.usage?.CPU || {};
    },
    ram() {
      return this.usage?.RAM || {};
    },
  },
  methods: {
    isWarning(value) {
      const int = value?.trim()?.replace('%', '');
      console.log();
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