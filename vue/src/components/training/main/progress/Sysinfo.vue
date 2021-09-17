<template>
  <div class="t-sysinfo" v-if="isEmptyAll()">
    <div class="t-sysinfo__label">Информация об устройстве</div>
    <div class="t-sysinfo__grid">
      <div v-if="isEmpty(gpu)">
        <div class="t-sysinfo__grid--item">GPU</div>
        <div :class="['t-sysinfo__grid--item', { warning: isWarning(gpu.gpu_utilization) }]">
          <p class="t-sysinfo__gpu-name">NVIDIA GeForce GTX 1060 6 GB</p>
          <p>{{ `${gpu.gpu_utilization || '0%'} (${gpu.gpu_memory_used || '0'} / ${gpu.gpu_memory_total || '0'})` }}</p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (gpu.gpu_utilization || 0) + '%' }"></div>
          </div>
        </div>
      </div>
      <div v-if="isEmpty(cpu)">
        <div class="t-sysinfo__grid--item">CPU</div>
        <div :class="['t-sysinfo__grid--item', { warning: isWarning(cpu.cpu_utilization) }]">
          <p>{{ `${cpu.cpu_utilization || '0%'} (${cpu.cpu_memory_used || '0'} / ${cpu.cpu_memory_total || '0'})` }}</p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (cpu.cpu_utilization || 0) + '%' }"></div>
          </div>
        </div>
      </div>
      <div v-if="isEmpty(ram)">
        <div class="t-sysinfo__grid--item">RAM</div>
        <div :class="['t-sysinfo__grid--item', { warning: isWarning(ram.ram_utilization) }]">
          <p>
            {{
              `${ram.ram_utilization + '% ' || '0%'} (${ram.ram_memory_used || '0'} / ${ram.ram_memory_total || '0'})`
            }}
          </p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (ram.ram_utilization || 0) + '%' }"></div>
          </div>
        </div>
      </div>
      <div v-if="isEmpty(disk)">
        <div class="t-sysinfo__grid-item">Disk</div>
        <div :class="['t-sysinfo__grid-item', { warning: isWarning(disk.disk_utilization) }]">
          <p>
            {{
              `${disk.disk_utilization + '% ' || '0%'} (${disk.disk_memory_used || '0'} / ${
                disk.disk_memory_total || '0'
              })`
            }}
          </p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (disk.disk_utilization || 0) + '%' }"></div>
          </div>
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
    isEmptyAll() {
      return this.isEmpty(this.disk) && (this.isEmpty(this.gpu) || this.isEmpty(this.cpu)) && this.isEmpty(this.ram)
    },
    isEmpty(obj) {
      return Object.keys(obj).length !== 0;
    },
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
  gap: 12px;
  font-size: 14px;
  width: 100%;
  p {
    color: #fff;
  }
  &__gpu-name {
    color: #fff !important;
  }
  &__grid {
    &--item {
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