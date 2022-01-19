<template>
  <div class="t-sysinfo">
    <div class="t-sysinfo__label">Информация об устройстве</div>
    <div class="t-sysinfo__grid">
      <div v-if="GPU.gpu_utilization">
        <div class="t-sysinfo__grid--item">GPU</div>
        <div :class="['t-sysinfo__grid--item', isWarning(GPU.gpu_utilization)]">
          <p class="t-sysinfo__gpu-name">{{ GPU.gpu_name }}</p>
          <p>{{ `${GPU.gpu_utilization}% (${GPU.gpu_memory_used} / ${GPU.gpu_memory_total})` }}</p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (GPU.gpu_utilization) + '%' }"></div>
          </div>
        </div>
      </div>
      <div v-if="CPU.cpu_utilization">
        <div class="t-sysinfo__grid--item">CPU</div>
        <div :class="['t-sysinfo__grid--item', isWarning(CPU.cpu_utilization)]">
          <p>{{`${CPU.cpu_utilization || 0}%`}}</p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: `${CPU.cpu_utilization || 0}%` }"></div>
          </div>
        </div>
      </div>
      <div>
        <div class="t-sysinfo__grid--item">RAM</div>
        <div :class="['t-sysinfo__grid--item',isWarning(RAM.ram_utilization) ]">
          <p>{{`${RAM.ram_utilization}% (${RAM.ram_memory_used} / ${RAM.ram_memory_total})`}}</p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (RAM.ram_utilization) + '%' }"></div>
          </div>
        </div>
      </div>
      <div>
        <div class="t-sysinfo__grid-item">Disk</div>
        <div :class="['t-sysinfo__grid-item', isWarning(Disk.disk_utilization)]">
          <p>{{`${Disk.disk_utilization + '%'} (${Disk.disk_memory_used} / ${Disk.disk_memory_total})`}}
          </p>
          <div class="t-sysinfo__progress-bar">
            <div class="t-sysinfo__progress-bar--fill" :style="{ width: (Disk.disk_utilization) + '%' }"></div>
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
    Disk: {
      type: Object,
      default: () => {
        return {
          disk_utilization: 0,
          disk_memory_total: 0,
          disk_memory_used: 0,
        }
      },
    },
    GPU: {
      type: Object,
      default: () => {
        return {
          gpu_utilization: 0,
          gpu_memory_total: 0,
          gpu_memory_used: 0,
        }
      },
    },
    CPU: {
      type: Object,
      default: () => {
        return {
          cpu_utilization: 0,
          cpu_memory_total: 0,
          cpu_memory_used: 0,
        }
      },
    },
    RAM: {
      type: Object,
      default: () => {
        return {
          ram_utilization: 0,
          ram_memory_total: 0,
          ram_memory_used: 0,
        }
      },
    },
  },
  methods: {
    isEmptyAll() {
      return this.isEmpty(this.Disk) && (this.isEmpty(this.GPU) || this.isEmpty(this.CPU)) && this.isEmpty(this.RAM);
    },
    isEmpty(obj) {
      return obj ? Object.keys(obj).length !== 0 : false;
    },
    isWarning(value) {
      const int = value ? +value : 0;
      return int > 80 ? 'warning' : int > 50 ? 'info' : '' ;
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
      &.info p {
        color: #ffc82c;
      }
      &.info div div {
        background: #ffc82c;
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