<template>
  <div class="t-time">
    <div class="t-time__train">
      <div class="t-time__timer-wrapper">
        <div class="t-time__timer">
          <span>Расчетное время обучения</span>
          <div>{{ formatTime(estimated_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Прошло времени</span>
          <div>{{ formatTime(elapsed_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Время до окончания обучения</span>
          <div>{{ formatTime(still_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Эпоха</span>
          <div>{{ epoch.current || 0 }}</div>
          <i>/</i>
          <div>{{ epoch.total || 0 }}</div>
        </div>
      </div>
      <div class="t-time__progress-bar">
        <div class="t-time__progress-bar--fill" :style="total(epoch)"></div>
      </div>
    </div>
    <div class="t-time__age">
      <div class="t-time__timer-wrapper">
        <div class="t-time__timer">
          <span>Среднее время эпохи</span>
          <div>{{ formatTime(avg_epoch_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Прошло времени на эпоху</span>
          <div>{{ formatTime(elapsed_epoch_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Время до окончания текущей эпохи</span>
          <div>{{ formatTime(still_epoch_time) }}</div>
        </div>
        <div class="t-time__timer">
          <span>Батч</span>
          <div>{{ batch.current || 0 }}</div>
          <i>/</i>
          <div>{{ batch.total || 0 }}</div>
        </div>
      </div>
      <div class="t-time__progress-bar">
        <div class="t-time__progress-bar--fill" :style="total(batch)"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-train-time',
  props: {
    avg_epoch_time: {
      type: Number,
      default: 0,
    },
    elapsed_epoch_time: {
      type: Number,
      default: 0,
    },
    elapsed_time: {
      type: Number,
      default: 0,
    },
    estimated_time: {
      type: Number,
      default: 0,
    },
    still_epoch_time: {
      type: Number,
      default: 0,
    },
    still_time: {
      type: Number,
      default: 0,
    },
    batch: {
      type: Object,
      default: () => {
        return { current: 0, total: 0 };
      },
    },
    epoch: {
      type: Object,
      default: () => {
        return { current: 0, total: 0 };
      },
    },
  },
  methods: {
    total({ total, current }) {
      let int = Math.round(current * 100  / total)
      int = isNaN(int) ? 0 : int
      if (int > 100 ) {
        int = 100
      }
      return { width: int + '%'};
    },
    formatTime(sec) {
      return `${Math.floor(sec / 3600)}h : ${Math.floor(sec % 3600 / 60)}m : ${Math.floor(sec % 3600 % 60)}s`;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-time {
  width: 100%;
  &__timer-wrapper {
    display: flex;
    gap: 20px;
    span {
      white-space: nowrap;
    }
  }
  &__progress-bar {
    background: #2b5278;
    border-radius: 5px;
    height: 10px;
    margin-top: 10px;
    &--fill {
      background: #65b9f4;
      height: 100%;
      border-radius: 5px;
    }
  }
  &__age {
    margin-top: 20px;
  }
  &__timer {
    i {
      font-style: normal;
      transform: translateY(25%);
    }
    &:first-child div {
      border: 1px solid #65b9f4;
    }
    &:nth-child(2) div {
      border: 1px solid #fff;
    }
    &:last-child {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      text-align: center;
      justify-content: space-between;
      flex-shrink: 0;
      flex-grow: 0;
      width: min-content;
      span {
        width: 92px;
        text-align: center;
      }
      div {
        width: 40px;
        text-align: center;
        padding: 8px 0;
      }
    }
    div {
      font-size: 14px;
      color: #ffffff;
      padding: 8px 10px;
      background: #242f3d;
      border: 1px solid #6c7883;
      border-radius: 10px;
      width: 212px;
      text-align: center;
      margin-top: 10px;
    }
  }
}
</style>